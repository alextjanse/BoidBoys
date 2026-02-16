struct Boid {
  position: vec4<f32>,
  velocity: vec4<f32>,
};

@binding(0) @group(0) var<storage, read_write> boids: array<Boid>;

// Workgroup shared memory for tile-based processing
var<workgroup> shared_boids: array<Boid, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
  let idx = global_id.x;
  let local_idx = local_id.x;
  let total = arrayLength(&boids);

  let separation_dist = 25.0;
  let align_dist = 50.0;
  let cohesion_dist = 50.0;
  
  let max_speed = 5.0;
  let max_force = 0.1;
  
  let separation_weight = 1.5;
  let alignment_weight = 1.0;
  let cohesion_weight = 1.0;

  let world_max = vec3<f32>(1000.0, 600.0, 600.0);
  let margin = 50.0;
  let turn_factor = 0.2;

  var my_pos = vec3<f32>(0.0);
  var my_vel = vec3<f32>(0.0);

  var sep_sum = vec3<f32>(0.0);
  var align_sum = vec3<f32>(0.0);
  var coh_sum = vec3<f32>(0.0);
  var count_sep: u32 = 0u;
  var count_align: u32 = 0u;
  var count_coh: u32 = 0u;

  if (idx < total) {
    my_pos = boids[idx].position.xyz;
    my_vel = boids[idx].velocity.xyz;
  }
  
  let num_tiles = (total + 255u) / 256u;
  for (var tile = 0u; tile < num_tiles; tile = tile + 1u) {
    let tile_start = tile * 256u;
    let global_neighbor_idx = tile_start + local_idx;
    
    // All threads cooperatively load a tile into shared memory
    if (global_neighbor_idx < total) {
      shared_boids[local_idx] = boids[global_neighbor_idx];
    }
    workgroupBarrier(); // ALL threads sync here
    
    // Only valid threads do the computation, but all still reach barrier
    if (idx < total) {
      for (var j = 0u; j < 256u; j = j + 1u) {
        let other_global_idx = tile_start + j;
        if (other_global_idx >= total || other_global_idx == idx) { continue; }
        
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
    workgroupBarrier(); // ALL threads sync here before next tile
  }

  // Only valid threads continue with steering and updates
  if (idx < total) {
    var accel = vec3<f32>(0.0);

    // Separation
    if (count_sep > 0u) {
      let desired = normalize(sep_sum / f32(count_sep)) * max_speed;
      var steer = desired - my_vel;
      accel += clamp(steer, vec3<f32>(-max_force), vec3<f32>(max_force)) * separation_weight;
    }

    // Alignment
    if (count_align > 0u) {
      let desired = normalize(align_sum / f32(count_align)) * max_speed;
      var steer = desired - my_vel;
      accel += clamp(steer, vec3<f32>(-max_force), vec3<f32>(max_force)) * alignment_weight;
    }

    // Cohesion
    if (count_coh > 0u) {
      let center = coh_sum / f32(count_coh);
      let desired = normalize(center - my_pos) * max_speed;
      var steer = desired - my_vel;
      accel += clamp(steer, vec3<f32>(-max_force), vec3<f32>(max_force)) * cohesion_weight;
    }

    var wall_accel = vec3<f32>(0.0);
    
    // X-axis check
    if (my_pos.x < margin) { 
      wall_accel.x += turn_factor; 
    } else if (my_pos.x > world_max.x - margin) { 
      wall_accel.x -= turn_factor; 
    }
    
    // Y-axis check
    if (my_pos.y < margin) { 
      wall_accel.y += turn_factor; 
    } else if (my_pos.y > world_max.y - margin) { 
      wall_accel.y -= turn_factor; 
    }

    // Z-axis check
    if (my_pos.z < margin) { 
      wall_accel.z += turn_factor; 
    } else if (my_pos.z > world_max.z - margin) { 
      wall_accel.z -= turn_factor; 
    }

    var new_vel = my_vel + accel + wall_accel;

    // Clamp speed to max_speed
    if (length(new_vel) > max_speed) {
      new_vel = normalize(new_vel) * max_speed;
    }

    var new_pos = my_pos + new_vel;
    new_pos = clamp(new_pos, vec3<f32>(0.0), world_max);

    boids[idx].position = vec4<f32>(new_pos, 1.0);
    boids[idx].velocity = vec4<f32>(new_vel, 0.0);
  }
}