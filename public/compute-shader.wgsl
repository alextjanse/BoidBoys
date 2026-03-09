// ============================================================
// Boids + Stigmergy Compute Shader — Spatial Hash Pipeline
// ============================================================

struct Boid {
  position: vec4<f32>,
  velocity: vec4<f32>,
};

struct SceneParams {
  separation_dist: f32,   // 0
  align_dist: f32,        // 4
  cohesion_dist: f32,     // 8
  max_speed: f32,         // 12
  max_force: f32,         // 16
  separation_weight: f32, // 20
  alignment_weight: f32,  // 24
  cohesion_weight: f32,   // 28
  margin: f32,            // 32
  turn_factor: f32,       // 36
  cell_size: f32,         // 40
  mode: f32,              // 44  (0=boids, 1=stigmergy)
  world_max: vec4<f32>,   // 48  (align 16)
  grid_dim: vec4<f32>,    // 64  (.w = num_cells)
  sensor_angle: f32,      // 80
  sensor_dist: f32,       // 84
  deposit_amount: f32,    // 88
  decay_rate: f32,        // 92
  diffusion_rate: f32,    // 96
  steer_strength: f32,    // 100
  random_strength: f32,   // 104
  frame_count: f32,       // 108
  phero_grid_dim: vec4<f32>, // 112 (.w = phero_num_cells)
};
// Total: 128 bytes = 32 floats

// ---- Resources ----
@binding(0) @group(0) var<storage, read_write> boids: array<Boid>;
@binding(1) @group(0) var<uniform> params: SceneParams;
@binding(2) @group(0) var<storage, read_write> cell_heads: array<atomic<i32>>;
@binding(3) @group(0) var<storage, read_write> boid_next: array<i32>;
@binding(4) @group(0) var<storage, read_write> matrices: array<f32>;
@binding(5) @group(0) var<storage, read_write> pheromone_src: array<f32>;
@binding(6) @group(0) var<storage, read_write> pheromone_dst: array<f32>;
@binding(7) @group(0) var<storage, read_write> deposit_buf: array<atomic<u32>>;

// ---- Helpers ----

fn cell_to_index(cell: vec3<i32>) -> i32 {
  let gd = vec3<i32>(vec3<f32>(params.grid_dim.xyz));
  return cell.x + cell.y * gd.x + cell.z * gd.x * gd.y;
}

fn pos_to_cell(pos: vec3<f32>) -> vec3<i32> {
  let cell = vec3<i32>(floor(pos / params.cell_size));
  let gd = vec3<i32>(vec3<f32>(params.grid_dim.xyz));
  return clamp(cell, vec3<i32>(0), gd - vec3<i32>(1));
}

fn phero_cell_to_index(cell: vec3<i32>) -> i32 {
  let pgd = vec3<i32>(vec3<f32>(params.phero_grid_dim.xyz));
  return cell.x + cell.y * pgd.x + cell.z * pgd.x * pgd.y;
}

fn pos_to_phero_cell(pos: vec3<f32>) -> vec3<i32> {
  let pgd = vec3<f32>(params.phero_grid_dim.xyz);
  let effective_cs = params.world_max.xyz / pgd;
  let cell = vec3<i32>(floor(pos / effective_cs));
  return clamp(cell, vec3<i32>(0), vec3<i32>(pgd) - vec3<i32>(1));
}

fn compute_wall_accel(pos: vec3<f32>) -> vec3<f32> {
  let margin = params.margin;
  let tf = params.turn_factor;
  let wm = params.world_max.xyz;
  var wa = vec3<f32>(0.0);
  if (pos.x < margin)          { wa.x += tf; }
  else if (pos.x > wm.x - margin) { wa.x -= tf; }
  if (pos.y < margin)          { wa.y += tf; }
  else if (pos.y > wm.y - margin) { wa.y -= tf; }
  if (pos.z < margin)          { wa.z += tf; }
  else if (pos.z > wm.z - margin) { wa.z -= tf; }
  return wa;
}

fn pcg_hash(input: u32) -> u32 {
  var state = input * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn rand_f32(seed: u32) -> f32 {
  return f32(pcg_hash(seed)) / 4294967295.0;
}

// ============================================================
// PHASE 1 — Spatial Hash Boids
// ============================================================

// Pass 1: Clear cell heads to -1 (empty)
@compute @workgroup_size(256)
fn clear_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let nc = u32(params.grid_dim.w);
  if (idx < nc) {
    atomicStore(&cell_heads[idx], -1);
  }
}

// Pass 2: Insert boids into spatial hash via linked list
@compute @workgroup_size(256)
fn hash_insert(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = arrayLength(&boids);
  if (idx >= total) { return; }

  let pos = boids[idx].position.xyz;
  let cell = pos_to_cell(pos);
  let ci = cell_to_index(cell);

  let old_head = atomicExchange(&cell_heads[ci], i32(idx));
  boid_next[idx] = old_head;
}

// Pass 3: Update boids using spatial-hash neighbor lookup
@compute @workgroup_size(256)
fn update_boids(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = arrayLength(&boids);
  if (idx >= total) { return; }

  let separation_dist = params.separation_dist;
  let align_dist      = params.align_dist;
  let cohesion_dist   = params.cohesion_dist;
  let max_speed       = params.max_speed;
  let max_force       = params.max_force;
  let sep_w           = params.separation_weight;
  let ali_w           = params.alignment_weight;
  let coh_w           = params.cohesion_weight;

  let my_pos = boids[idx].position.xyz;
  let my_vel = boids[idx].velocity.xyz;
  let my_cell = pos_to_cell(my_pos);
  let gd = vec3<i32>(vec3<f32>(params.grid_dim.xyz));

  var sep_sum   = vec3<f32>(0.0);
  var align_sum = vec3<f32>(0.0);
  var coh_sum   = vec3<f32>(0.0);
  var cnt_s = 0u;
  var cnt_a = 0u;
  var cnt_c = 0u;

  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let nc = my_cell + vec3<i32>(dx, dy, dz);
        if (any(nc < vec3<i32>(0)) || any(nc >= gd)) { continue; }
        let ci = cell_to_index(nc);
        var cur = atomicLoad(&cell_heads[ci]);
        var iters = 0u;
        while (cur != -1 && iters < 512u) {
          let oi = u32(cur);
          if (oi != idx) {
            let op = boids[oi].position.xyz;
            let ov = boids[oi].velocity.xyz;
            let d = distance(my_pos, op);
            if (d < separation_dist && d > 0.0) {
              sep_sum += normalize(my_pos - op) / d;
              cnt_s++;
            }
            if (d < align_dist && d > 0.0) {
              align_sum += ov;
              cnt_a++;
            }
            if (d < cohesion_dist && d > 0.0) {
              coh_sum += op;
              cnt_c++;
            }
          }
          cur = boid_next[oi];
          iters++;
        }
      }
    }
  }

  var accel = vec3<f32>(0.0);
  if (cnt_s > 0u) {
    let desired = normalize(sep_sum / f32(cnt_s)) * max_speed;
    accel += clamp(desired - my_vel, vec3<f32>(-max_force), vec3<f32>(max_force)) * sep_w;
  }
  if (cnt_a > 0u) {
    let desired = normalize(align_sum / f32(cnt_a)) * max_speed;
    accel += clamp(desired - my_vel, vec3<f32>(-max_force), vec3<f32>(max_force)) * ali_w;
  }
  if (cnt_c > 0u) {
    let center = coh_sum / f32(cnt_c);
    let desired = normalize(center - my_pos) * max_speed;
    accel += clamp(desired - my_vel, vec3<f32>(-max_force), vec3<f32>(max_force)) * coh_w;
  }

  let wall_accel = compute_wall_accel(my_pos);
  var new_vel = my_vel + accel + wall_accel;
  if (length(new_vel) > max_speed) { new_vel = normalize(new_vel) * max_speed; }
  let new_pos = clamp(my_pos + new_vel, vec3<f32>(0.0), params.world_max.xyz);

  boids[idx].position = vec4<f32>(new_pos, 1.0);
  boids[idx].velocity = vec4<f32>(new_vel, 0.0);
}

// Pass 4: Compute 4x4 instance matrices for Three.js
@compute @workgroup_size(256)
fn compute_matrices(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = arrayLength(&boids);
  if (idx >= total) { return; }

  let pos = boids[idx].position.xyz;
  let vel = boids[idx].velocity.xyz;
  let speed = length(vel);

  var forward: vec3<f32>;
  var right:   vec3<f32>;
  var up:      vec3<f32>;

  if (speed > 0.001) {
    forward = vel / speed;
    let world_up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(dot(forward, world_up)) > 0.99) {
      right = normalize(cross(vec3<f32>(1.0, 0.0, 0.0), forward));
    } else {
      right = normalize(cross(world_up, forward));
    }
    up = cross(forward, right);
  } else {
    right   = vec3<f32>(1.0, 0.0, 0.0);
    up      = vec3<f32>(0.0, 1.0, 0.0);
    forward = vec3<f32>(0.0, 0.0, 1.0);
  }

  // Column-major 4x4 for Three.js
  let b = idx * 16u;
  matrices[b +  0u] = right.x;
  matrices[b +  1u] = right.y;
  matrices[b +  2u] = right.z;
  matrices[b +  3u] = 0.0;
  matrices[b +  4u] = up.x;
  matrices[b +  5u] = up.y;
  matrices[b +  6u] = up.z;
  matrices[b +  7u] = 0.0;
  matrices[b +  8u] = forward.x;
  matrices[b +  9u] = forward.y;
  matrices[b + 10u] = forward.z;
  matrices[b + 11u] = 0.0;
  matrices[b + 12u] = pos.x;
  matrices[b + 13u] = pos.y;
  matrices[b + 14u] = pos.z;
  matrices[b + 15u] = 1.0;
}

// ============================================================
// PHASE 2 — Stigmergy
// ============================================================

// Pass S1: Clear pheromone deposit accumulator
@compute @workgroup_size(256)
fn clear_deposit(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let nc = u32(params.phero_grid_dim.w);
  if (idx < nc) {
    atomicStore(&deposit_buf[idx], 0u);
  }
}

// Pass S2: Each boid deposits pheromone at its position
@compute @workgroup_size(256)
fn deposit_pheromone(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = arrayLength(&boids);
  if (idx >= total) { return; }

  let pos = boids[idx].position.xyz;
  let cell = pos_to_phero_cell(pos);
  let ci = phero_cell_to_index(cell);
  let amount = u32(params.deposit_amount * 1024.0);
  atomicAdd(&deposit_buf[ci], amount);
}

// Pass S3: Diffuse + decay pheromone field
// Reads pheromone_src (binding 5) + deposit_buf, writes pheromone_dst (binding 6)
@compute @workgroup_size(256)
fn diffuse_decay(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let num_pc = u32(params.phero_grid_dim.w);
  if (idx >= num_pc) { return; }

  let pgd = vec3<i32>(vec3<f32>(params.phero_grid_dim.xyz));

  // Flat index -> 3D cell coords
  let cz = i32(idx) / (pgd.x * pgd.y);
  let rem = i32(idx) - cz * pgd.x * pgd.y;
  let cy = rem / pgd.x;
  let cx = rem - cy * pgd.x;

  let center_val = pheromone_src[idx];
  let deposit_val = f32(atomicLoad(&deposit_buf[idx])) / 1024.0;

  // 6-face-neighbor diffusion
  var nsum = 0.0;
  var ncnt = 0.0;

  let c = vec3<i32>(cx, cy, cz);
  if (cx > 0)         { nsum += pheromone_src[u32(cell_to_phero_idx(c + vec3<i32>(-1, 0, 0), pgd))]; ncnt += 1.0; }
  if (cx < pgd.x - 1) { nsum += pheromone_src[u32(cell_to_phero_idx(c + vec3<i32>( 1, 0, 0), pgd))]; ncnt += 1.0; }
  if (cy > 0)         { nsum += pheromone_src[u32(cell_to_phero_idx(c + vec3<i32>( 0,-1, 0), pgd))]; ncnt += 1.0; }
  if (cy < pgd.y - 1) { nsum += pheromone_src[u32(cell_to_phero_idx(c + vec3<i32>( 0, 1, 0), pgd))]; ncnt += 1.0; }
  if (cz > 0)         { nsum += pheromone_src[u32(cell_to_phero_idx(c + vec3<i32>( 0, 0,-1), pgd))]; ncnt += 1.0; }
  if (cz < pgd.z - 1) { nsum += pheromone_src[u32(cell_to_phero_idx(c + vec3<i32>( 0, 0, 1), pgd))]; ncnt += 1.0; }

  var avg = 0.0;
  if (ncnt > 0.0) { avg = nsum / ncnt; }
  let with_deposit = center_val + deposit_val;
  let diffused = mix(with_deposit, avg, params.diffusion_rate);
  pheromone_dst[idx] = max(diffused * params.decay_rate, 0.0);
}

// helper for diffuse_decay — pheromone 3D→flat index
fn cell_to_phero_idx(cell: vec3<i32>, pgd: vec3<i32>) -> i32 {
  return cell.x + cell.y * pgd.x + cell.z * pgd.x * pgd.y;
}

// helper: sample pheromone at a world position from pheromone_dst
fn sample_pheromone(pos: vec3<f32>) -> f32 {
  let clamped = clamp(pos, vec3<f32>(0.0), params.world_max.xyz - vec3<f32>(0.01));
  let cell = pos_to_phero_cell(clamped);
  let ci = phero_cell_to_index(cell);
  if (ci < 0 || u32(ci) >= u32(params.phero_grid_dim.w)) { return 0.0; }
  return pheromone_dst[ci];
}

// Pass S4: Stigmergy boid update — sense pheromone, steer, separation, wall avoidance
@compute @workgroup_size(256)
fn stigmergy_update(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = arrayLength(&boids);
  if (idx >= total) { return; }

  let my_pos = boids[idx].position.xyz;
  let my_vel = boids[idx].velocity.xyz;
  let speed  = length(my_vel);
  let max_speed = params.max_speed;
  let max_force = params.max_force;

  // --- Sensor directions ---
  var forward = vec3<f32>(0.0, 0.0, 1.0);
  if (speed > 0.001) {
    forward = my_vel / speed;
  }
  let world_up = vec3<f32>(0.0, 1.0, 0.0);
  var right_dir: vec3<f32>;
  if (abs(dot(forward, world_up)) > 0.99) {
    right_dir = normalize(cross(vec3<f32>(1.0, 0.0, 0.0), forward));
  } else {
    right_dir = normalize(cross(world_up, forward));
  }

  let sa = params.sensor_angle;
  let sd = params.sensor_dist;
  let cos_a = cos(sa);
  let sin_a = sin(sa);

  let left_dir  = forward * cos_a - right_dir * sin_a;
  let right_sen = forward * cos_a + right_dir * sin_a;

  let left_pos   = my_pos + left_dir  * sd;
  let center_pos = my_pos + forward   * sd;
  let right_pos  = my_pos + right_sen * sd;

  let pl = sample_pheromone(left_pos);
  let pc = sample_pheromone(center_pos);
  let pr = sample_pheromone(right_pos);

  // Steer toward strongest pheromone
  var desired_dir = forward;
  if (pc >= pl && pc >= pr) {
    desired_dir = forward;
  } else if (pl > pr) {
    desired_dir = left_dir;
  } else {
    desired_dir = right_sen;
  }

  // Random perturbation
  let seed = idx * 1000u + u32(params.frame_count);
  let rx = rand_f32(seed) * 2.0 - 1.0;
  let ry = rand_f32(seed + 1u) * 2.0 - 1.0;
  let rz = rand_f32(seed + 2u) * 2.0 - 1.0;
  let rand_vec = vec3<f32>(rx, ry, rz) * params.random_strength;

  var new_forward = normalize(mix(forward, desired_dir, params.steer_strength) + rand_vec);
  var new_vel = new_forward * max_speed;

  // --- Separation via spatial hash ---
  let my_cell = pos_to_cell(my_pos);
  let gd = vec3<i32>(vec3<f32>(params.grid_dim.xyz));
  var sep_sum = vec3<f32>(0.0);
  var cnt_s = 0u;

  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let nc = my_cell + vec3<i32>(dx, dy, dz);
        if (any(nc < vec3<i32>(0)) || any(nc >= gd)) { continue; }
        let ci = cell_to_index(nc);
        var cur = atomicLoad(&cell_heads[ci]);
        var iters = 0u;
        while (cur != -1 && iters < 512u) {
          let oi = u32(cur);
          if (oi != idx) {
            let d = distance(my_pos, boids[oi].position.xyz);
            if (d < params.separation_dist && d > 0.0) {
              sep_sum += normalize(my_pos - boids[oi].position.xyz) / d;
              cnt_s++;
            }
          }
          cur = boid_next[oi];
          iters++;
        }
      }
    }
  }

  if (cnt_s > 0u) {
    let desired = normalize(sep_sum / f32(cnt_s)) * max_speed;
    let sep_accel = clamp(desired - new_vel, vec3<f32>(-max_force), vec3<f32>(max_force)) * params.separation_weight;
    new_vel += sep_accel;
  }

  // Wall avoidance
  new_vel += compute_wall_accel(my_pos);

  // Clamp speed
  if (length(new_vel) > max_speed) { new_vel = normalize(new_vel) * max_speed; }
  let new_pos = clamp(my_pos + new_vel, vec3<f32>(0.0), params.world_max.xyz);

  boids[idx].position = vec4<f32>(new_pos, 1.0);
  boids[idx].velocity = vec4<f32>(new_vel, 0.0);
}
