using System.Numerics;

namespace BoidBoys.Models;

public class Boid
{
  public Vector3 Position { get; set; }
  public Vector3 Velocity { get; set; }
  public Vector3 Acceleration { get; set; }

  public Boid(Vector3 position, Vector3 velocity)
  {
    Position = position;
    Velocity = velocity;
    Acceleration = Vector3.Zero;
  }

  public void ApplyForce(Vector3 force)
  {
    Acceleration += force;
  }

  public void Update()
  {
    Velocity += Acceleration;
    Velocity = Vector3.Clamp(Velocity, -new Vector3(5), new Vector3(5)); // Max speed
    Position += Velocity;
    Acceleration = Vector3.Zero;
  }

  public void WrapAround(float width, float height, float depth)
  {
    if (Position.X < 0) Position = Position with { X = width };
    if (Position.X > width) Position = Position with { X = 0 };
    if (Position.Y < 0) Position = Position with { Y = height };
    if (Position.Y > height) Position = Position with { Y = 0 };
    if (Position.Z < 0) Position = Position with { Z = depth };
    if (Position.Z > depth) Position = Position with { Z = 0 };
  }
}
