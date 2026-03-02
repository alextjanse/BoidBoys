struct Boid {
  position: vec4<f32>,
  velocity: vec4<f32>,
};

struct KDNode {
  boid_idx: i32,
  split_axis: i32,
  left: i32,
  right: i32,
};

@binding(0) @group(0) var<storage, read_write> boids: array<Boid>;
@binding(1) @group(0) var<storage, read> kd_tree: array<KDNode>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let total = arrayLength(&boids);

  if (idx >= total) { return; }

  let separation_dist = 25.0;
  let align_dist = 50.0;
  let cohesion_dist = 50.0;
  let max_speed = 5.0;
  let max_force = 0.1;
  let separation_weight = 1.5;
  let alignment_weight = 1.0;
  let cohesion_weight = 0.5;
  let world_max = vec3<f32>(1000.0, 600.0, 600.0);
  let margin = 100.0;
  let turn_factor = 0.2;
  let search_radius = max(separation_dist, max(align_dist, cohesion_dist));

  let my_pos = boids[idx].position.xyz;
  let my_vel = boids[idx].velocity.xyz;

  var sep_sum = vec3<f32>(0.0);
  var align_sum = vec3<f32>(0.0);
  var coh_sum = vec3<f32>(0.0);
  var count_sep = 0u;
  var count_align = 0u;
  var count_coh = 0u;

  // ----- k-d tree range search (iterative, stack-based) -----
  var stack: array<i32, 64>;
  var stack_top: i32 = 1;
  stack[0] = 0; // push root

  while (stack_top > 0) {
    stack_top--;
    let node_idx = stack[stack_top];

    if (node_idx < 0) { continue; }

    let node = kd_tree[u32(node_idx)];
    let other_idx = u32(node.boid_idx);
    let other_pos = boids[other_idx].position.xyz;
    let other_vel = boids[other_idx].velocity.xyz;

    // Check this node's boid for neighbor interactions
    if (other_idx != idx) {
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

    // Determine which subtrees to visit
    let split_val = other_pos[u32(node.split_axis)];
    let query_val = my_pos[u32(node.split_axis)];
    let diff = query_val - split_val;

    // Always visit the near subtree; visit the far subtree only when the
    // search sphere crosses the splitting plane.
    if (diff <= 0.0) {
      // query point lies on the left side
      if (node.left >= 0 && stack_top < 64) {
        stack[stack_top] = node.left;
        stack_top++;
      }
      if (abs(diff) < search_radius && node.right >= 0 && stack_top < 64) {
        stack[stack_top] = node.right;
        stack_top++;
      }
    } else {
      // query point lies on the right side
      if (node.right >= 0 && stack_top < 64) {
        stack[stack_top] = node.right;
        stack_top++;
      }
      if (diff < search_radius && node.left >= 0 && stack_top < 64) {
        stack[stack_top] = node.left;
        stack_top++;
      }
    }
  }

  // ----- apply flocking forces -----
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

  // ----- wall avoidance -----
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