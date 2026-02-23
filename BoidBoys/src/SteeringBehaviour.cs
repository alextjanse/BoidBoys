using System.Numerics;

namespace BoidBoys.src
{
    public class SteeringBehaviour
    {
        public Vector3 CalculateTotalVelocity(Boid[] boids, int selfIndex, List<int> neighbours)
        {
            if (neighbours.Count == 0)
            {
                return Vector3.Zero;
            }

            Vector3 total = Vector3.Zero;

            total += CalculateSeparation(boids, selfIndex, neighbours);
            total += CalculateAlignment(boids, neighbours);
            total += CalculateCohesion(boids, neighbours);

            return total;
        }

        /// <summary>
        /// Steer to avoid crowding local flockmates 
        /// </summary>
        public Vector3 CalculateSeparation(Boid[] boids, int selfIndex, List<int> neighbours)
        {
            ref readonly Boid self = ref boids[selfIndex];

            Vector3 totalVelocity = Vector3.Zero;

            for (int i = 0; i < neighbours.Count; i++)
            {
                ref readonly Boid other = ref boids[neighbours[i]];
                totalVelocity += self.position - other.position;
            }

            return -totalVelocity;
        }

        /// <summary>
        /// Steer towards the average heading of local flockmates 
        /// </summary>
        public Vector3 CalculateAlignment(Boid[] boids, List<int> neighbours)
        {
            Vector3 averageVelocity = Vector3.Zero;

            for (int i = 0; i < neighbours.Count; i++)
            {
                averageVelocity += boids[neighbours[i]].velocity;
            }

            averageVelocity /= neighbours.Count;
            return averageVelocity;
        }

        /// <summary>
        /// Steer to move toward the average position of local flockmates 
        /// </summary>
        public Vector3 CalculateCohesion(Boid[] boids, List<int> neighbours)
        {
            Vector3 averagePosition = Vector3.Zero;

            for (int i = 0; i < neighbours.Count; i++)
            {
                averagePosition += boids[neighbours[i]].position;
            }

            averagePosition /= neighbours.Count;
            return averagePosition;
        }
    }
}