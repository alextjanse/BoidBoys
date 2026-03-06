struct Boid {
  position: vec4<f32>,
  velocity: vec4<f32>,
};

@binding(0) @group(0) var<storage, read_write> boids: array<Boid>;

struct SceneParams {
  separation_dist: f32,
  align_dist: f32,
  cohesion_dist: f32,
  max_speed: f32,
  max_force: f32,
  separation_weight: f32,
  alignment_weight: f32,
  cohesion_weight: f32,
  margin: f32,  
  turn_factor: f32,
  padding0: f32,
  padding1: f32,
  world_max: vec4<f32>,
};

@binding(1) @group(0) var<uniform> params: SceneParams;

// Workgroup shared memory for tile-based processing
var<workgroup> shared_boids: array<Boid, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let idx = global_id.x;
  let local_idx = local_id.x;
  let total = arrayLength(&boids);

  // Read scene/global parameters from the uniform buffer
  let separation_dist = params.separation_dist;
  let align_dist = params.align_dist;
  let cohesion_dist = params.cohesion_dist;
  let max_speed = params.max_speed;
  let max_force = params.max_force;
  let separation_weight = params.separation_weight;
  let alignment_weight = params.alignment_weight;
  let cohesion_weight = params.cohesion_weight;
  let world_max = params.world_max.xyz;
  let margin = params.margin;
  let turn_factor = params.turn_factor;

  var my_pos: vec3<f32>;
  var my_vel: vec3<f32>;

  if (idx < total) {
    my_pos = boids[idx].position.xyz;
    my_vel = boids[idx].velocity.xyz;
  }

  var sep_sum = vec3<f32>(0.0);
  var align_sum = vec3<f32>(0.0);
  var coh_sum = vec3<f32>(0.0);
  var count_sep = 0u;
  var count_align = 0u;
  var count_coh = 0u;

  let num_tiles = (total + 255u) / 256u;
  for (var tile = 0u; tile < num_tiles; tile++) {
    let tile_start = tile * 256u;
    
    // Cooperative load into shared memory
    if (tile_start + local_idx < total) {
      shared_boids[local_idx] = boids[tile_start + local_idx];
    }
    workgroupBarrier();

    if (idx < total) {
      for (var j = 0u; j < 256u; j++) {
        let other_idx = tile_start + j;
        if (other_idx >= total || other_idx == idx) { continue; }

        let other_pos = shared_boids[j].position.xyz;
        let other_vel = shared_boids[j].velocity.xyz;
        let dist = distance(my_pos, other_pos);

        if (dist < separation_dist && dist > 0.0) {
          sep_sum += (normalize(my_pos - other_pos) / dist);
          count_sep++;
        }
        if (dist < align_dist && dist > 0.0) {
          align_sum += other_vel;
          count_align++;
        }
        if (dist < cohesion_dist && dist > 0.0) {
          coh_sum += other_pos;
          count_coh++;
        }
      }
    }
    workgroupBarrier();
  }

  if (idx < total) {
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

    var wall_accel = vec3<f32>(0.0);
    if (my_pos.x < margin) { wall_accel.x += turn_factor; } 
    else if (my_pos.x > world_max.x - margin) { wall_accel.x -= turn_factor; }
    if (my_pos.y < margin) { wall_accel.y += turn_factor; } 
    else if (my_pos.y > world_max.y - margin) { wall_accel.y -= turn_factor; }
    if (my_pos.z < margin) { wall_accel.z += turn_factor; } 
    else if (my_pos.z > world_max.z - margin) { wall_accel.z -= turn_factor; }

    var new_vel = my_vel + accel + wall_accel;
    if (length(new_vel) > max_speed) { new_vel = normalize(new_vel) * max_speed; }
    
    let new_pos = clamp(my_pos + new_vel, vec3<f32>(0.0), world_max);

    boids[idx].position = vec4<f32>(new_pos, 1.0);
    boids[idx].velocity = vec4<f32>(new_vel, 0.0);
  }
}