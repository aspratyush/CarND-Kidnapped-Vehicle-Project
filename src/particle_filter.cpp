/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static std::default_random_engine gen;
static double EPSILON = 0.00001;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  if (is_initialized == true){
    std::cout<<"Particles have already been initialized!!!\n";
  }
  else{
      //1. set the number of particles to use
      num_particles = 100;

      //2a. create gaussian distributions for x, y, theta to sample from
      std::normal_distribution<double> dst_x(x, std[0]);
      std::normal_distribution<double> dst_y(y, std[1]);
      std::normal_distribution<double> dst_theta(theta, std[2]);

      //3. iterate over the number and create particles
      for(int i = 0; i < num_particles; i++){
          // create a particle
          Particle particle;
          // set the particle attributes
          particle.id = i;
          particle.x = dst_x(gen);
          particle.y = dst_y(gen);
          particle.theta = dst_theta(gen);
          // set weight to 1
          particle.weight = 1.0;

          // add this particle to the particles vector in the class
          particles.push_back(particle);
          // add the weight to the weight vector
          //weights.push_back(particle.weight);
      }

      //4. set is_initialized variable
      is_initialized = true;
      std::cout<<num_particles<<" particles initialized...\n";
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  //1b. create gaussian distributions to sample noise from
  std::normal_distribution<double> noise_x(0, std_pos[0]);
  std::normal_distribution<double> noise_y(0, std_pos[1]);
  std::normal_distribution<double> noise_theta(0, std_pos[2]);

  //----------------- USING BICYCLE MODEL----------------------//
  // reference : Lesson 12 (Motion Models), Video#3 : Yaw Rate & Velocity
  //2. iterate over the particles and predict position based on yaw rate
  for(std::size_t i = 0; i < particles.size(); i++){
      if (fabs(yaw_rate) < EPSILON){
          particles[i].x += velocity*delta_t*cos(particles[i].theta);
          particles[i].y += velocity*delta_t*sin(particles[i].theta);
      }
      else{
          particles[i].x += (velocity/yaw_rate)*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
          particles[i].y += (velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta+yaw_rate*delta_t));
          particles[i].theta += yaw_rate*delta_t;
      }

      //add noise
      particles[i].x += noise_x(gen);
      particles[i].y += noise_y(gen);
      particles[i].theta += noise_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

  //1. iterate over the observations
  for (size_t i = 0; i < observations.size(); i++){

    // initialize min_distance and index
    //reference: https://stackoverflow.com/questions/8690567/setting-an-int-to-infinity-in-c#8690584
    double min_distance = std::numeric_limits<double>::max();
    int index = -1;

    for (size_t j = 0; j < predicted.size(); j++){
      double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      // store the id if distance lesser than threshold
      if (distance < min_distance){
        min_distance = distance;
        index = predicted[j].id;
      }
    }

    // update the index of predicted landmark
    observations[i].id = index;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs>& observations, const Map& map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    
    //1. Iterate over the particles
    for (int i = 0; i < num_particles; i++){        
        
        //2. Filter the landmarks within the particles range
        std::vector<LandmarkObs> filtered_landmarks;
        for(size_t j = 0; j < map_landmarks.landmark_list.size(); j++){
            
            //2a. find the distance between particle and landmark
            double distance = dist(particles[i].x, particles[i].y, 
                                    map_landmarks.landmark_list[j].x_f, 
                                    map_landmarks.landmark_list[j].y_f);
            
            //2b. check range and add to vector list           
            if (distance <= sensor_range){
                filtered_landmarks.push_back(LandmarkObs{ map_landmarks.landmark_list[j].id_i, 
                                                        map_landmarks.landmark_list[j].x_f, 
                                                        map_landmarks.landmark_list[j].y_f });
            }
        }

        //3. Transform co-ordinates from car to map
        std::vector<LandmarkObs> transformed_observations;
        for (size_t j = 0; j < observations.size(); j++){
            //3a. use the co-ordinate transformation equations taught in class
            double x_m = cos(particles[i].theta)*observations[j].x - sin(particles[i].theta)*observations[j].y + particles[i].x;
            double y_m = sin(particles[i].theta)*observations[j].x + cos(particles[i].theta)*observations[j].y + particles[i].y;
            
            //3b. add the transformed co-ordinates to the vector
            transformed_observations.push_back(LandmarkObs{ observations[j].id, x_m, y_m });
        }

        //4. Associate the transformed observations to the filtered landmarks
        dataAssociation(filtered_landmarks, transformed_observations);

        //5. Reset the weight for this iteration
        particles[i].weight = 1.0;
        
        //6. Calculate the new weights
        for (size_t j = 0; j < transformed_observations.size(); j++){

            double landmarkX = 0.0, landmarkY = 0.0;            
            //6a. find the matching id of the filtered_landmark
            for (size_t k = 0; k < filtered_landmarks.size(); k++){
                if (filtered_landmarks[k].id == transformed_observations[j].id) {
                  landmarkX = filtered_landmarks[k].x;
                  landmarkY = filtered_landmarks[k].y;
                  break;
                }
            }
            
            double d_x = transformed_observations[j].x - landmarkX;
            double d_y = transformed_observations[j].y - landmarkY;

            double weight = ( 1/(2*M_PI*std_landmark[0]*std_landmark[1])) * 
                            exp( -( d_x*d_x/(2*std_landmark[0]*std_landmark[0]) + 
                                (d_y*d_y/(2*std_landmark[1]*std_landmark[1])) ) );
            
            //6b. add a check for weight - add a small weight if zero
            if (weight == 0) {
                particles[i].weight *= EPSILON;
            }
            else {
                particles[i].weight *= weight;
            }
        }
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // creating a new vector weights instead of using the class provided one -- to auto clear
    std::vector<double> weights;
    //1. Add particle weights ti weight vector and get max weight
    double max_weight = std::numeric_limits<double>::min();
    for (int i = 0; i < num_particles; i++){

        //1a. add the particles weight to weight vector
        weights.push_back(particles[i].weight);

        //1b. also store the maximum weight
        if (particles[i].weight > max_weight){
            max_weight = particles[i].weight;
        }
    }

    //2. Creating uniform distributions to sample index
    std::uniform_real_distribution<double> dist_double(0.0, max_weight);
    std::uniform_int_distribution<int> dist_index(0, num_particles - 1);

    //2a. get an index from the distribution
    int index = dist_index(gen);

    //2b. calculate beta using wheel sampling taught in class
    double beta = 0.0;
    std::vector<Particle> resampled_particles;
    //2c. iterate over all the particles
    for(int i = 0; i < num_particles; i++){
        // sample a beta from the distribution
        beta += dist_double(gen) * 2.0;
        //run a loop till weights[index] > beta
        while (beta > weights[index]){
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        resampled_particles.push_back(particles[index]);
    }
    
    //3. reset the particles
    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
		             const std::vector<double>& sense_x, const std::vector<double>& sense_y){
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
