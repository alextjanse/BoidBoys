// ============================================================
// Spatial-Hashing Boid Simulation  –  Counting Sort Pipeline
// ============================================================
// Pipeline order per frame (5 passes total, any boid count):
//   1. clear_counts      – zero cell_count array
//   2. count_boids       – atomicAdd per cell to count boids
//   3. prefix_sum        – sequential prefix sum → cell_start / cell_end
//   4. scatter           – scatter boids into sorted order via atomicAdd
//   5. update_boids      – neighbour search via spatial hash
// ============================================================

// ───── Structs ─────
struct Boid {
  position: vec4<f32>,
  velocity: vec4<f32>,
};

struct HashEntry {
  cell_hash: u32,
  boid_index: u32,
};

// ───── Constants ─────
const CELL_SIZE: f32 = 50.0;
const GRID_X: u32 = 50u;   // ceil(5000 / 50)
const GRID_Y: u32 = 50u;   // ceil(5000 / 50)
const GRID_Z: u32 = 50u;   // ceil(5000 / 50)
const NUM_CELLS: u32 = GRID_X * GRID_Y * GRID_Z; // 2880

// ───── Bindings ─────
// Each entry point only references the subset it needs;
// auto-layout will create per-pipeline bind-group layouts.
@group(0) @binding(0) var<storage, read_write> boids:        array<Boid>;
@group(0) @binding(1) var<storage, read_write> hash_entries: array<HashEntry>;
@group(0) @binding(2) var<storage, read_write> cell_count:   array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> cell_start:   array<u32>;
@group(0) @binding(4) var<storage, read_write> cell_end:     array<atomic<u32>>;

// ───── Helpers ─────
fn compute_cell_hash(pos: vec3<f32>) -> u32 {
  let cx = min(u32(max(floor(pos.x / CELL_SIZE), 0.0)), GRID_X - 1u);
  let cy = min(u32(max(floor(pos.y / CELL_SIZE), 0.0)), GRID_Y - 1u);
  let cz = min(u32(max(floor(pos.z / CELL_SIZE), 0.0)), GRID_Z - 1u);
  return cx + cy * GRID_X + cz * GRID_X * GRID_Y;
}

// ═══════════════════════════════════════════════════════════════
// Pass 1 – Clear cell counts  (dispatched over NUM_CELLS)
// ═══════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn clear_counts(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= NUM_CELLS) { return; }
  atomicStore(&cell_count[idx], 0u);
}

// ═══════════════════════════════════════════════════════════════
// Pass 2 – Count boids per cell  (dispatched over BOID_COUNT)
// ═══════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn count_boids(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= arrayLength(&boids)) { return; }
  let hash = compute_cell_hash(boids[idx].position.xyz);
  atomicAdd(&cell_count[hash], 1u);
}

// ═══════════════════════════════════════════════════════════════
// Pass 3 – Exclusive prefix sum  (single thread – 2880 iterations)
//          Writes cell_start and initialises cell_end for scatter.
// ═══════════════════════════════════════════════════════════════
@compute @workgroup_size(1)
fn prefix_sum() {
  var running = 0u;
  for (var i = 0u; i < NUM_CELLS; i++) {
    let cnt = atomicLoad(&cell_count[i]);
    cell_start[i] = running;
    atomicStore(&cell_end[i], running); // scatter will atomicAdd from here
    running += cnt;
  }
}

// ═══════════════════════════════════════════════════════════════
// Pass 4 – Scatter boids into sorted order  (dispatched over BOID_COUNT)
//          After this pass, cell_end[c] == cell_start[c] + cell_count[c].
// ═══════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= arrayLength(&boids)) { return; }
  let hash = compute_cell_hash(boids[idx].position.xyz);
  let slot = atomicAdd(&cell_end[hash], 1u);
  hash_entries[slot] = HashEntry(hash, idx);
}

// ═══════════════════════════════════════════════════════════════
// Pass 5 – Update boid positions using spatial-hash neighbour search
//          (dispatched over BOID_COUNT)
//          Uses squared distances to eliminate all sqrt from inner loop.
// ═══════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn update_boids(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx   = gid.x;
  let total = arrayLength(&boids);
  if (idx >= total) { return; }

  // ── Boid parameters ──
  let sep_dist_sq       = 625.0;   // 25²
  let align_dist_sq     = 2500.0;  // 50²
  let cohesion_dist_sq  = 2500.0;  // 50²
  let max_speed         = 5.0;
  let max_force         = 0.1;
  let separation_weight = 1.5;
  let alignment_weight  = 1.0;
  let cohesion_weight   = 0.5;
  let world_max = vec3<f32>(5000.0, 5000.0, 5000.0);
  let margin      = 100.0;
  let turn_factor = 0.2;

  let my_pos = boids[idx].position.xyz;
  let my_vel = boids[idx].velocity.xyz;

  let my_cell = vec3<i32>(
    clamp(i32(floor(my_pos.x / CELL_SIZE)), 0, i32(GRID_X) - 1),
    clamp(i32(floor(my_pos.y / CELL_SIZE)), 0, i32(GRID_Y) - 1),
    clamp(i32(floor(my_pos.z / CELL_SIZE)), 0, i32(GRID_Z) - 1),
  );

  var sep_sum   = vec3<f32>(0.0);
  var align_sum = vec3<f32>(0.0);
  var coh_sum   = vec3<f32>(0.0);
  var count_sep   = 0u;
  var count_align = 0u;
  var count_coh   = 0u;

  // Search 3×3×3 neighbourhood (27 cells)
  for (var dz: i32 = 0; dz <= 0; dz++) {
    for (var dy: i32 = 0; dy <= 0; dy++) {
      for (var dx: i32 = 0; dx <= 0; dx++) {
        let nx = my_cell.x + dx;
        let ny = my_cell.y + dy;
        let nz = my_cell.z + dz;

        if (nx < 0 || ny < 0 || nz < 0 ||
            nx >= i32(GRID_X) || ny >= i32(GRID_Y) || nz >= i32(GRID_Z)) { continue; }

        let cell = u32(nx) + u32(ny) * GRID_X + u32(nz) * GRID_X * GRID_Y;
        let start = cell_start[cell];
        let end_val = atomicLoad(&cell_end[cell]);
        if (start >= end_val) { continue; }

        for (var s = start; s < end_val; s++) {
          let other_idx = hash_entries[s].boid_index;
          if (other_idx == idx) { continue; }

          let other_pos = boids[other_idx].position.xyz;
          let diff = my_pos - other_pos;
          let dist_sq = dot(diff, diff);

          // Separation: normalize(diff)/dist = diff/dist² = diff/dist_sq
          if (dist_sq < sep_dist_sq && dist_sq > 0.0) {
            sep_sum += diff / dist_sq;
            count_sep++;
          }
          if (dist_sq < align_dist_sq && dist_sq > 0.0) {
            align_sum += boids[other_idx].velocity.xyz;
            count_align++;
          }
          if (dist_sq < cohesion_dist_sq && dist_sq > 0.0) {
            coh_sum += other_pos;
            count_coh++;
          }
        }
      }
    }
  }

  // ── Apply steering forces ──
  var accel = vec3<f32>(0.0);

  if (count_sep > 0u) {
    let desired = normalize(sep_sum / f32(count_sep)) * max_speed;
    accel += clamp(desired - my_vel, vec3<f32>(-max_force), vec3<f32>(max_force)) * separation_weight;
  }
  if (count_align > 0u) {
    let desired = normalize(align_sum / f32(count_align)) * max_speed;
    accel += clamp(desired - my_vel, vec3<f32>(-max_force), vec3<f32>(max_force)) * alignment_weight;
  }
  if (count_coh > 0u) {
    let center = coh_sum / f32(count_coh);
    let desired = normalize(center - my_pos) * max_speed;
    accel += clamp(desired - my_vel, vec3<f32>(-max_force), vec3<f32>(max_force)) * cohesion_weight;
  }

  // ── Wall avoidance ──
  var wall_accel = vec3<f32>(0.0);
  if (my_pos.x < margin) { wall_accel.x += turn_factor; }
  else if (my_pos.x > world_max.x - margin) { wall_accel.x -= turn_factor; }
  if (my_pos.y < margin) { wall_accel.y += turn_factor; }
  else if (my_pos.y > world_max.y - margin) { wall_accel.y -= turn_factor; }
  if (my_pos.z < margin) { wall_accel.z += turn_factor; }
  else if (my_pos.z > world_max.z - margin) { wall_accel.z -= turn_factor; }

  // ── Integrate ──
  var new_vel = my_vel + accel + wall_accel;
  if (length(new_vel) > max_speed) { new_vel = normalize(new_vel) * max_speed; }

  let new_pos = clamp(my_pos + new_vel, vec3<f32>(0.0), world_max);

  boids[idx].position = vec4<f32>(new_pos, 1.0);
  boids[idx].velocity = vec4<f32>(new_vel, 0.0);
}