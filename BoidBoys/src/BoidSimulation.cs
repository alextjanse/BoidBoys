using System.Numerics;

namespace BoidBoys.src
{
    public class BoidSimulation
    {
        private readonly ThreadLocal<Random> rnd = new(() => new Random(Guid.NewGuid().GetHashCode()));

        private readonly SpatialHash hash;
        private readonly float neighbourRadius = 5f;

        // reusable buffers (no GC)
        private Vector3[] positionBuffer = [];
        private readonly SteeringBehaviour steering = new();

        public BoidSimulation(int maxBoids, float cellSize)
        {
            hash = new SpatialHash(cellSize, maxBoids);
        }

        public Boid[] Initialize(int count, Vector3 boundingSize)
        {
            Boid[] boids = new Boid[count];

            Parallel.For(0, count, i =>
            {
                var r = rnd.Value!;
                boids[i].position = RandomVec3(r) * boundingSize;
                boids[i].velocity = Vector3.Zero;
            });

            return boids;
        }

        public Boid[] Step(Boid[] prev)
        {
            int boidCount = prev.Length;
            Boid[] boids = (Boid[])prev.Clone();

            if (positionBuffer.Length < boidCount)
            {
                positionBuffer = new Vector3[boidCount];
            }

            for (int i = 0; i < boidCount; i++)
            {
                positionBuffer[i] = prev[i].position;
            }

            hash.Build(positionBuffer);

            Parallel.For(0, boidCount, i =>
            {
                List<int> localNeighbours = hash.Query(positionBuffer, prev[i].position, neighbourRadius);

                Vector3 steer = steering.CalculateTotalVelocity(prev, i, localNeighbours);

                boids[i].velocity += steer;
                boids[i].position += boids[i].velocity;
            });

            return boids;
        }

        private static Vector3 RandomVec3(Random r)
        {
            return new Vector3(
                RandomMinPlus1(r),
                RandomMinPlus1(r),
                RandomMinPlus1(r));
        }

        private static float RandomMinPlus1(Random r)
        {
            return r.NextSingle() * 2f - 1f;
        }
    }
}