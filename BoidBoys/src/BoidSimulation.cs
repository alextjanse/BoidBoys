using System.Numerics;

namespace BoidBoys.src
{
    public class BoidSimulation
    {
        private readonly Random rnd = new();
        public Boid[] Initialize(Vector3 boundingSize)
        {
            Boid[] boids = [];
            Parallel.For(0, boids.Length, i =>
            {
                boids[i].position = RandomVec3() * boundingSize;
            });
            return boids;
        }

        public Boid[] Step(Boid[] prev)
        {
            Boid[] boids = [.. prev];
            Parallel.For(0, boids.Length, i =>
            {
                // boids[i].velocity += new SteeringBehaviour().CalculateTotalVelocity(boids[i], /*TODO: neighbours*/);
                boids[i].position += boids[i].velocity;
            });
            return boids;
        }

        private Vector3 RandomVec3()
        {
            return new Vector3(RandomMinPlus1(), RandomMinPlus1(), RandomMinPlus1());
        }

        private float RandomMinPlus1()
        {
            return rnd.NextSingle() * 2f - 1f;
        }
    }
}