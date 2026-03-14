// #region Boids compute shader — spatial hash pipeline
// Boids compute shader — spatial hash pipeline

// #region Data structures
// Boid storage layout: position.xyzw and velocity.xyzw
struct Boid {
  position: vec4<f32>,
  velocity: vec4<f32>,
};

// Uniform scene parameters (20 floats = 80 bytes)
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
  _pad: f32,              // 44  (padding for vec4 alignment)
  world_max: vec4<f32>,   // 48  (align 16)
  grid_dim: vec4<f32>,    // 64  (.w = num_cells)
};
// Total: 80 bytes = 20 floats
// #endregion

// #region GPU resources

// GPU resources: storage buffers and uniforms
@binding(0) @group(0) var<storage, read_write> boids: array<Boid>;
@binding(1) @group(0) var<uniform> params: SceneParams;
@binding(2) @group(0) var<storage, read_write> cell_heads: array<atomic<i32>>;
@binding(3) @group(0) var<storage, read_write> boid_next: array<i32>;
@binding(4) @group(0) var<storage, read_write> matrices: array<f32>;

// #endregion

// #region Helper functions
// Helper functions for spatial hash and bounds handling
fn cell_to_index(cell: vec3<i32>) -> i32 {
  let gd = vec3<i32>(vec3<f32>(params.grid_dim.xyz));
  return cell.x + cell.y * gd.x + cell.z * gd.x * gd.y;
}

fn pos_to_cell(pos: vec3<f32>) -> vec3<i32> {
  let cell = vec3<i32>(floor(pos / params.cell_size));
  let gd = vec3<i32>(vec3<f32>(params.grid_dim.xyz));
  return clamp(cell, vec3<i32>(0), gd - vec3<i32>(1));
}

// Compute acceleration to keep boids away from world bounds
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

// #endregion

// #region Compute passes
// Pass 1: clear cell heads to -1 (empty)
@compute @workgroup_size(256)
fn clear_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let nc = u32(params.grid_dim.w);
  if (idx < nc) {
    atomicStore(&cell_heads[idx], -1);
  }
}

// Pass 2: insert boids into the spatial hash using a linked list
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

// Pass 3: update boids using spatial-hash neighbor lookup
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

  // Search neighboring cells and accumulate neighbor contributions
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

  // Combine separation, alignment and cohesion contributions into a single acceleration
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
// #endregion
