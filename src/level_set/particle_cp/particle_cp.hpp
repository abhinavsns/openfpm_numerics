//
// Created by lschulze on 2021-10-12
//
/**
 * @file particle_cp.hpp
 * @class particle_cp_redistancing
 *
 * @brief Class for reinitializing a level-set function into a signed distance function using the Closest-Point
 * method on particles.
 *
 * @details The redistancing scheme here is based on Saye, Robert. "High-order methods for computing distances
 * to implicitly defined surfaces." Communications in Applied Mathematics and Computational Science 9.1 (2014):
 * 107-141.
 * Within this file, it has been extended so that it also works on arbitrarily distributed particles. It mainly
 * consists of these steps: Firstly, the particle distribution is assessed and a subset is determined, which will
 * provide a set of sample points. For these, an interpolation of the old signed-distance function values in their
 * support yields a polynomial whose zero level-set reconstructs the surface. The interpolation polynomials are
 * then used to create a collection of sample points on the interface. Finally, a Newton-optimisation is performed
 * in order to determine the closest point on the surface to a given query point, allowing for an update of the
 * signed distance function.
 *
 *
 * @author Lennart J. Schulze
 * @date October 2021
 */

#include <math.h>
#include "Vector/vector_dist.hpp"
#include "DCPSE/Dcpse.hpp"
#include "DCPSE/MonomialBasis.hpp"
#include "Draw/DrawParticles.hpp"
#include "Operators/Vector/vector_dist_operators.hpp"
#include <chrono>
#include "regression/regression.hpp"

template<unsigned int dim, unsigned int n_c> using particles_surface = vector_dist<dim, double, aggregate<int, int, double, double[dim], double[n_c], EVectorXd>>;
struct Redist_options
{
	size_t max_iter = 1000;	// params for the Newton algorithm for the solution of the constrained
	double tolerance = 1e-11; // optimization problem, and for the projection of the sample points on the interface
	
	double H; // interparticle spacing
	double r_cutoff_factor; // radius of neighbors to consider during interpolation, as factor of H
	double sampling_radius;

	int compute_normals = 0;
	int compute_curvatures = 0;

	int write_sdf = 1;
	int write_cp = 0;

	unsigned int minter_poly_degree = 4;
	float minter_lp_degree = 1.0;

	int verbose = 0;
	int min_num_particles = 0; // if 0, cutoff radius is used, else the int is the minimum number of particles for the regression.
				   // If min_num_particles is used, the cutoff radius is used for the cell list. It is sensible to use
				   // a smaller rcut for the celllist, to promote symmetric supports (otherwise it is possible that the
				   // particle is at the corner of a cell and hence only has neighbors in certain directions)
	float r_cutoff_factor_min_num_particles; // this is the rcut for the celllist
	int only_narrowband = 1; // only redistance particles with phi < sampling_radius, or all particles if only_narrowband = 0
};

// Add new polynomial degrees here in case new interpolation schemes are implemented.
// Template typenames for initialization of vd_s
struct quadratic {};
struct bicubic {};
struct taylor4 {};
struct minter_polynomial {};


template <typename particles_in_type, typename polynomial_degree, size_t phi_field, size_t closest_point_field, size_t normal_field, size_t curvature_field, unsigned int num_minter_coeffs>
class particle_cp_redistancing
{
public:
	particle_cp_redistancing(particles_in_type & vd, Redist_options &redistOptions) : redistOptions(redistOptions),
					  	  	  	  	  	  	  vd_in(vd),
											  vd_s(vd.getDecomposition(), 0),
											  r_cutoff2(redistOptions.r_cutoff_factor*redistOptions.r_cutoff_factor*redistOptions.H*redistOptions.H),
											  minterModelpcp(RegressionModel<dim, vd_s_sdf>(
                                                                                          redistOptions.minter_poly_degree, redistOptions.minter_lp_degree))
	{
		// initialize polynomial degrees
		if (std::is_same<polynomial_degree,quadratic>::value) polynomialDegree = "quadratic";
		else if (std::is_same<polynomial_degree,bicubic>::value) polynomialDegree = "bicubic";
		else if (std::is_same<polynomial_degree,taylor4>::value) polynomialDegree = "taylor4";
		else if (std::is_same<polynomial_degree,minter_polynomial>::value) polynomialDegree = "minterpolation";
		else throw std::invalid_argument("Invalid polynomial degree given. Valid choices currently are quadratic, bicubic, taylor4, minter_polynomial.");
	}
	
	void run_redistancing()
	{
		if (redistOptions.verbose) 
		{
			std::cout<<"Verbose mode. Make sure the vd.getProp<4>(a) is an integer that pcp can write surface flags onto."<<std::endl; 
			std::cout<<"Minterpol variable is "<<minterpol<<std::endl;
		}

		detect_surface_particles();
		
		interpolate_sdf_field();

		find_closest_point();
	}

private:
	static constexpr size_t num_neibs = 0;
	static constexpr size_t vd_s_close_part = 1;
	static constexpr size_t vd_s_sdf = 2;
	static constexpr size_t vd_s_sample = 3;
	static constexpr size_t interpol_coeff = 4;
	static constexpr size_t minter_coeff = 5;
	// static constexpr size_t vd_s_minter_model = 5;
	static constexpr size_t vd_in_sdf = phi_field; // this is really required in the vd_in vector, so users need to know about it.
	static constexpr size_t vd_in_close_part = 4; // this is not needed by the method, but more for debugging purposes, as it shows all particles for which
						      // interpolation and sampling is performed.
	static constexpr size_t vd_in_normal = normal_field;
	static constexpr size_t vd_in_curvature = curvature_field;
	static constexpr size_t	vd_in_cp = closest_point_field;

	Redist_options redistOptions;
	static constexpr unsigned int minterpol = (num_minter_coeffs > 0) ? 1 : 0;
	particles_in_type &vd_in;
	static constexpr unsigned int dim = particles_in_type::dims;
	static constexpr unsigned int n_c = (minterpol == 1) ? num_minter_coeffs :
	                                    (dim == 2 && std::is_same<polynomial_degree,quadratic>::value) ? 6 :
										(dim == 2 && std::is_same<polynomial_degree,bicubic>::value) ? 16 :
										(dim == 2 && std::is_same<polynomial_degree,taylor4>::value) ? 15 :
										(dim == 3 && std::is_same<polynomial_degree,taylor4>::value) ? 35 : 1;

	int dim_r = dim;
	int n_c_r = n_c;

	particles_surface<dim, n_c> vd_s;
	double r_cutoff2;
	std::string polynomialDegree;
	RegressionModel<dim, vd_s_sdf> minterModelpcp;

	int return_sign(double phi)
	{
		if (phi > 0) return 1;
		if (phi < 0) return -1;
		return 0;
	}

	// this function lays the groundwork for the interpolation step. It classifies particles according to two possible classes:
	// (a)	close particles: These particles are close to the surface and will be the central particle in the subsequent interpolation step.
	//		After the interpolation step, they will also be projected on the surface. The resulting position on the surface will be referred to as a sample point,
	//		which will provide the initial guesses for the final Newton optimization, which finds the exact closest-point towards any given query point.
	//		Further, the number of neighbors in the support of close particles are stored, to efficiently initialize the Vandermonde matrix in the interpolation.
	// (b)	surface particles: These particles have particles in their support radius, which are close particles (a). In order to compute the interpolation
	//		polynomials for (a), these particles need to be stored. An alternative would be to find the relevant particles in the original particle vector.

	void detect_surface_particles()
	{
		vd_in.template ghost_get<vd_in_sdf>();

		auto NN = vd_in.getCellList(sqrt(r_cutoff2) + redistOptions.H);
		auto part = vd_in.getDomainIterator();

		while (part.isNext())
		{
			vect_dist_key_dx akey = part.get();
            		// depending on the application this can spare computational effort
            		if ((redistOptions.only_narrowband) && (std::abs(vd_in.template getProp<vd_in_sdf>(akey)) > redistOptions.sampling_radius))
            		{
                		++part;
                		continue;
            		}
			int surfaceflag = 0;
			int sgn_a = return_sign(vd_in.template getProp<vd_in_sdf>(akey));
			Point<dim,double> xa = vd_in.getPos(akey);
			int num_neibs_a = 0;
			double min_sdf = abs(vd_in.template getProp<vd_in_sdf>(akey));
			vect_dist_key_dx min_sdf_key = akey;
			if (redistOptions.verbose) vd_in.template getProp<vd_in_close_part>(akey) = 0;
			int isclose = 0;

			auto Np = NN.template getNNIterator<NO_CHECK>(NN.getCell(xa));
			while (Np.isNext())
			{
				vect_dist_key_dx bkey = Np.get();
				int sgn_b = return_sign(vd_in.template getProp<vd_in_sdf>(bkey));
				Point<dim, double> xb = vd_in.getPos(bkey);
	            		Point<dim,double> dr = xa - xb;
	            		double r2 = norm2(dr);

	            		// check if the particle will provide a polynomial interpolation and a sample point on the interface
	            		if ((sqrt(r2) < (1.5*redistOptions.H)) && (sgn_a != sgn_b)) isclose = 1;

	            		// count how many particles are in the neighborhood and will be part of the interpolation
	            		if (r2 < r_cutoff2)	++num_neibs_a;

	            		// decide whether this particle will be required for any interpolation. This is the case if it has a different sign in its slightly extended neighborhood
	            		// its up for debate if this is stable or not. Possibly, this could be avoided by simply using vd_in in the interpolation step.
	            		if ((sqrt(r2) < ((redistOptions.r_cutoff_factor + 1.5)*redistOptions.H)) && (sgn_a != sgn_b)) surfaceflag = 1;
	            		++Np;
	            
			}

			if (surfaceflag) // these particles will play a role in the subsequent interpolation: Either they carry interpolation polynomials,
			{		 // or they will be taken into account in interpolation
				vd_s.add();
				for(int k = 0; k < dim; k++) vd_s.getLastPos()[k] = vd_in.getPos(akey)[k];
				vd_s.template getLastProp<vd_s_sdf>() = vd_in.template getProp<vd_in_sdf>(akey);
				vd_s.template getLastProp<num_neibs>() = num_neibs_a;

				if (isclose) // these guys will carry an interpolation polynomial and a resulting sample point
				{
					vd_s.template getLastProp<vd_s_close_part>() = 1;
					if (redistOptions.verbose) vd_in.template getProp<vd_in_close_part>(akey) = 1;
				}
				else // these particles will not carry an interpolation polynomial
				{
					vd_s.template getLastProp<vd_s_close_part>() = 0;
					if (redistOptions.verbose) vd_in.template getProp<vd_in_close_part>(akey) = 0;
				}
			}
			++part;
		}
	}
	
	void interpolate_sdf_field()
	{
		int message_insufficient_support = 0;
		int message_projection_fail = 0;

		vd_s.template ghost_get<vd_s_sdf>();
		double r_cutoff_celllist = sqrt(r_cutoff2);
		if (minterpol && (redistOptions.min_num_particles != 0)) r_cutoff_celllist = redistOptions.r_cutoff_factor_min_num_particles*redistOptions.H;
		auto NN_s = vd_s.getCellList(r_cutoff_celllist);
		auto part = vd_s.getDomainIterator();

		// iterate over particles that will get an interpolation polynomial and generate a sample point
		while (part.isNext())
		{
			vect_dist_key_dx a = part.get();
			
			// only the close particles (a) will get the full treatment (interpolation + projection)
			if (vd_s.template getProp<vd_s_close_part>(a) != 1)
			{
				++part;
				continue;
			}

			const int num_neibs_a = vd_s.template getProp<num_neibs>(a);
			Point<dim, double> xa = vd_s.getPos(a);
			int neib = 0;
            		int k_project = 0;
			// initialize monomial basis functions m, Vandermonde row builder vrb, Vandermonde matrix V, the right-hand
			// side vector phi and the solution vector that contains the coefficients of the interpolation polynomial, c.

			// The functions called here are originally from the DCPSE work. Minter regression is better suited for the application.
			if (!minterpol){
				MonomialBasis <dim> m(4);

				if (polynomialDegree == "quadratic") {
					unsigned int degrees[dim];
				for (int k = 0; k < dim; k++) degrees[k] = 1;
				m = MonomialBasis<dim>(degrees, 1);
				} else if (polynomialDegree == "taylor4") {
			    		unsigned int degrees[dim];
			    		for (int k = 0; k < dim; k++) degrees[k] = 1;
			    		unsigned int degrees_2 = 3;
			    		if (dim == 3) degrees_2 = 2;
			    		m = MonomialBasis<dim>(degrees, degrees_2);
				} else if (polynomialDegree == "bicubic") {
			    	// do nothing as it has already been initialized like this
				}

				if (m.size() > num_neibs_a) message_insufficient_support = 1;

				VandermondeRowBuilder<dim, double> vrb(m);

				EMatrix<double, Eigen::Dynamic, Eigen::Dynamic> V(num_neibs_a, m.size());
            			EMatrix<double, Eigen::Dynamic, 1> phi(num_neibs_a, 1);
            			EMatrix<double, Eigen::Dynamic, 1> c(m.size(), 1);

            			// iterate over the support of the central particle, in order to build the Vandermonde matrix and the
            			// right-hand side vector
            			auto Np = NN_s.template getNNIterator<NO_CHECK>(NN_s.getCell(vd_s.getPos(a)));

            			while (Np.isNext()) {
                			vect_dist_key_dx b = Np.get();
                			Point<dim, double> xb = vd_s.getPos(b);
                			Point<dim, double> dr = xa - xb;
                			double r2 = norm2(dr);

                			if (r2 < r_cutoff2) {
                    				// Fill phi-vector from the right hand side
        	            			phi[neib] = vd_s.template getProp<vd_s_sdf>(b);
               		     			vrb.buildRow(V, neib, xb, 1.0);

                    				++neib;
                			}
                			++Np;
            			}

            			// solve the system of equations using an orthogonal decomposition (This is a solid choice for an 
				// overdetermined system of equations with a bad conditioning, which is usually the case for the 
            			// Vandermonde matrices.
            			c = V.completeOrthogonalDecomposition().solve(phi);

            			// store the coefficients at the particle
            			for (int k = 0; k < m.size(); k++) {
                			vd_s.template getProp<interpol_coeff>(a)[k] = c[k];
            			}

				// after interpolation coefficients have been computed and stored, perform Newton-style projections
				// towards the interface in order to obtain a sample.
				// initialise derivatives of the interpolation polynomial and an interim position vector x. The iterations
				// start at the central particle x_a.

				// double incr; // this is an optional variable in case the iterations are aborted depending on the magnitude
						// of the increment instead of the magnitude of the polynomial value and the number of iterations.
				EMatrix<double, Eigen::Dynamic, 1> grad_p(dim_r, 1);
				double grad_p_mag2;

				EMatrix<double, Eigen::Dynamic, 1> x(dim_r, 1);
				for(int k = 0; k < dim; k++) x[k] = xa[k];
				double p = get_p(x, c, polynomialDegree);

				EMatrix<double, Eigen::Dynamic, 1> x_debug(dim_r, 1);
				for(int k = 0; k < dim_r; k++) x_debug[k] = 0.25;
				
				// do the projections.
				while ((abs(p) > redistOptions.tolerance) && (k_project < redistOptions.max_iter))
				{
					grad_p = get_grad_p(x, c, polynomialDegree);
                    			grad_p_mag2 = grad_p.dot(grad_p);

                    			x = x - p * grad_p / grad_p_mag2;
                    			p = get_p(x, c, polynomialDegree);
                    			//incr = sqrt(p*p*dpdx*dpdx/(grad_p_mag2*grad_p_mag2) + p*p*dpdy*dpdy/(grad_p_mag2*grad_p_mag2));
                    			++k_project;
                		}
                		// store the resulting sample point on the surface at the central particle.
                		for(int k = 0; k < dim; k++) vd_s.template getProp<vd_s_sample>(a)[k] = x[k];
           		 }

			if (minterpol)
			{
				if(redistOptions.min_num_particles == 0) 
				{
            				auto regSupport = RegressionSupport<decltype(vd_s), decltype(NN_s)>(vd_s, part, sqrt(r_cutoff2), RADIUS, NN_s);
					if (regSupport.getNumParticles() < n_c) message_insufficient_support = 1;
					minterModelpcp.computeCoeffs(vd_s, regSupport);
				}
				else 
				{
            				auto regSupport = RegressionSupport<decltype(vd_s), decltype(NN_s)>(vd_s, part, n_c + 3, AT_LEAST_N_PARTICLES, NN_s);
					if (regSupport.getNumParticles() < n_c) message_insufficient_support = 1;
					minterModelpcp.computeCoeffs(vd_s, regSupport);
				}

            			auto& minterModel = minterModelpcp.model;
				vd_s.template getProp<minter_coeff>(a) = minterModel->getCoeffs();

            			double grad_p_minter_mag2;

				EMatrix<double, Eigen::Dynamic, 1> grad_p_minter(dim_r, 1);
				EMatrix<double, Eigen::Dynamic, 1> x_minter(dim_r, 1);
				for(int k = 0; k < dim_r; k++) x_minter[k] = xa[k];

				double p_minter = get_p_minter(x_minter, minterModel);
				k_project = 0;
				while ((abs(p_minter) > redistOptions.tolerance) && (k_project < redistOptions.max_iter))
				{
					grad_p_minter = get_grad_p_minter(x_minter, minterModel);
					grad_p_minter_mag2 = grad_p_minter.dot(grad_p_minter);
					x_minter = x_minter - p_minter*grad_p_minter/grad_p_minter_mag2;
					p_minter = get_p_minter(x_minter, minterModel);
					//incr = sqrt(p*p*dpdx*dpdx/(grad_p_mag2*grad_p_mag2) + p*p*dpdy*dpdy/(grad_p_mag2*grad_p_mag2));
					++k_project;
            			}
				// store the resulting sample point on the surface at the central particle.
				for(int k = 0; k < dim; k++) vd_s.template getProp<vd_s_sample>(a)[k] = x_minter[k];
		    	}
			if (k_project == redistOptions.max_iter) 
			{
				if (redistOptions.verbose) std::cout<<"didnt work for "<<a.getKey()<<std::endl;
				message_projection_fail = 1;
			}

			++part;
		}

		if (message_insufficient_support) std::cout<<"Warning: less number of neighbours than required for interpolation"
   							<<" for some particles. Consider using at least N particles function."<<std::endl;
        	if (message_projection_fail) std::cout<<"Warning: Newton-style projections towards the interface do not satisfy"
                					<<" given tolerance for some particles"<<std::endl;

	}
	
	// This function now finds the exact closest point on the surface to any given particle.
	// For this, it iterates through all particles in the input vector (since all need to be redistanced),
	// and finds the closest sample point on the surface. Then, it uses this sample point as an initial guess
	// for a subsequent optimization problem, which finds the exact closest point on the surface.
	// The optimization problem is formulated as follows: Find the point in space with the minimal distance to
	// the given query particle, with the constraint that it needs to lie on the surface. It is realized with a
	// Lagrange multiplier that ensures that the solution lies on the surface, i.e. p(x)=0.
	// The polynomial that is used for checking if the constraint is fulfilled is the interpolation polynomial
	// carried by the sample point.

	void find_closest_point()
	{
		// iterate over all particles, i.e. do closest point optimisation for all particles, and initialize
		// all relevant variables.

		vd_s.template ghost_get<vd_s_close_part,vd_s_sample,interpol_coeff>();

		auto NN_s = vd_s.getCellList(redistOptions.sampling_radius);
		auto part = vd_in.getDomainIterator();

		int message_step_limitation = 0;
		int message_convergence_problem = 0;

		// do iteration over all query particles
		while (part.isNext())
		{
			vect_dist_key_dx a = part.get();
			
			if ((redistOptions.only_narrowband) && (std::abs(vd_in.template getProp<vd_in_sdf>(a)) > redistOptions.sampling_radius))
			{
				++part;
				continue;
			}

			// initialise all variables specific to the query particle
			EMatrix<double, Eigen::Dynamic, 1> xa(dim_r, 1);
			for(int k = 0; k < dim; k++) xa[k] = vd_in.getPos(a)[k];

			double val;
			// spatial variable that is optimized
			EMatrix<double, Eigen::Dynamic, 1> x(dim_r, 1);
			// Increment variable containing the increment of spatial variables + 1 Lagrange multiplier
			EMatrix<double, Eigen::Dynamic, 1> dx(dim_r + 1, 1);
			for(int k = 0; k < (dim + 1); k++) dx[k] = 1.0;

			// Lagrange multiplier lambda
			double lambda = 0.0;
			// values regarding the gradient of the Lagrangian and the Hessian of the Lagrangian, which contain
			// derivatives of the constraint function also.
			double p = 0;
			EMatrix<double, Eigen::Dynamic, 1> grad_p(dim_r, 1);
			for(int k = 0; k < dim; k++) grad_p[k] = 0.0;

			EMatrix<double, Eigen::Dynamic, Eigen::Dynamic> H_p(dim_r, dim_r);
			for(int k = 0; k < dim; k++)
			{
				for(int l = 0; l < dim; l++) H_p(k, l) = 0.0;
			}

			EMatrix<double, Eigen::Dynamic, 1> c(n_c_r, 1);
			for (int k = 0; k < n_c; k++) c[k] = 0.0;

			// f(x, lambda) is the Lagrangian, initialize its gradient and Hessian
			EMatrix<double, Eigen::Dynamic, 1> nabla_f(dim_r + 1, 1);
			EMatrix<double, Eigen::Dynamic, Eigen::Dynamic> H_f(dim_r + 1, dim_r + 1);
			for(int k = 0; k < (dim + 1); k++)
			{
				nabla_f[k] = 0.0;
				for(int l = 0; l < (dim + 1); l++) H_f(k, l) = 0.0;
			}

			// Initialise variables for searching the closest sample point. Note that this is not a very elegant
			// solution to initialise another x_a vector, but it is done because of the two different types EMatrix and
			// point vector, and can probably be made nicer.
			Point<dim, double> xaa = vd_in.getPos(a);

			// Now we will iterate over the sample points, which means iterating over vd_s.
			auto Np = NN_s.template getNNIterator<NO_CHECK>(NN_s.getCell(vd_in.getPos(a)));

			// distance is the the minimum distance to be beaten
			double distance = 1000000.0;
			// dist_calc is the variable for the distance that is computed per sample point
			double dist_calc = 1000000000.0;
			// b_min is the handle referring to the particle that carries the sample point with the minimum distance so far
			decltype(a) b_min = a;

			while (Np.isNext())
			{
				vect_dist_key_dx b = Np.get();

				// only go for particles (a) that carry sample points
				if (!vd_s.template getProp<vd_s_close_part>(b))
				{
					++Np;
					continue;
				}

				Point<dim, double> xbb = vd_s.template getProp<vd_s_sample>(b);

				// compute the distance towards the sample point and check if it the closest so far. If yes, store the new
				// closest distance to be beaten and store the handle
				dist_calc = abs(xbb.distance(xaa));

				if (dist_calc < distance)
				{
					distance = dist_calc;
					b_min = b;
				}
				++Np;
			}

			// set x0 to the sample point which was closest to the query particle
			for(int k = 0; k < dim; k++) x[k] = vd_s.template getProp<vd_s_sample>(b_min)[k];

			// these two variables are mainly required for optional subroutines in the optimization procedure and for
			// warnings if the solution strays far from the neighborhood.
			EMatrix<double, Eigen::Dynamic, 1> x00(dim_r,1);
			for(int k = 0; k < dim; k++) x00[k] = x[k];

			EMatrix<double, Eigen::Dynamic, 1> x00x(dim_r,1);
			for(int k = 0; k < dim; k++) x00x[k] = 0.0;

			// take the interpolation polynomial of the sample particle closest to the query particle
			for (int k = 0 ; k < n_c ; k++) c[k] = vd_s.template getProp<interpol_coeff>(b_min)[k];

            		auto& model = minterModelpcp.model;
		    	Point<dim, int> derivOrder;
			Point<dim, double> grad_p_minter;
			if (minterpol)
			{
				model->setCoeffs(vd_s.template getProp<minter_coeff>(b_min));
			}
			
			if(redistOptions.verbose)
			{
                		std::cout<<std::setprecision(16)<<"VERBOSE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% for particle "<<a.getKey()<<std::endl;
                		std::cout<<"x_poly: "<<vd_s.getPos(b_min)[0]<<", "<<vd_s.getPos(b_min)[1]<<"\nxa: "<<xa[0]<<", "<<xa[1]<<"\nx_0: "<<x[0]<<", "<<x[1]<<"\nc: "<<c<<std::endl;
                		if (!minterpol)
                		{
                    			std::cout<<"interpol_i(x_0) = "<<get_p(x, c, polynomialDegree)<<std::endl;
                		}
                		else
                		{
                    			std::cout<<"interpol_i(x_0) = "<<get_p_minter(x, model)<<std::endl;
                		}
			}

			// variable containing updated distance at iteration k
			EMatrix<double, Eigen::Dynamic, 1> xax = x - xa;
			// iteration counter k
			int k_newton = 0;

			// calculations needed specifically for k_newton == 0
			if (!minterpol)
			{
                		p = get_p(x, c, polynomialDegree);
                		grad_p = get_grad_p(x, c, polynomialDegree);
            		}

			if (minterpol)
            		{
				p = get_p_minter(x, model);
                		grad_p = get_grad_p_minter(x, model);
            		}

			// this guess for the Lagrange multiplier is taken from the original paper by Saye and can be done since
			// p is approximately zero at the sample point.
			lambda = -xax.dot(grad_p)/grad_p.dot(grad_p);

			for(int k = 0; k < dim; k++) nabla_f[k] = xax[k] + lambda*grad_p[k];
			nabla_f[dim] = p;

			nabla_f(Eigen::seq(0, dim - 1)) = xax + lambda*grad_p;
			nabla_f[dim] = p;
			// The exit criterion for the Newton optimization is on the magnitude of the gradient of the Lagrangian,
			// which is zero at a solution.
			double nabla_f_norm = nabla_f.norm();

			// Do Newton iterations.
			while((nabla_f_norm > redistOptions.tolerance) && (k_newton<redistOptions.max_iter))
			{
				// gather required derivative values
				if (!minterpol) H_p = get_H_p(x, c, polynomialDegree);
				if (minterpol) H_p = get_H_p_minter(x, model);

				// Assemble Hessian matrix, grad f has been computed at the end of the last iteration.
				H_f(Eigen::seq(0, dim_r - 1), Eigen::seq(0, dim_r - 1)) = lambda*H_p;
				for(int k = 0; k < dim; k++) H_f(k, k) = H_f(k, k) + 1.0;
				H_f(Eigen::seq(0, dim_r - 1), dim_r) = grad_p;
				H_f(dim_r, Eigen::seq(0, dim_r - 1)) = grad_p.transpose();
				H_f(dim_r, dim_r) = 0.0;

				// compute Newton increment
				dx = - H_f.inverse()*nabla_f;

				// prevent Newton algorithm from leaving the support radius by scaling step size. It is invoked in case 
				// the increment exceeds 50% of the cutoff radius and simply scales the step length down to 10% until it
				// does not exceed 50% of the cutoff radius.

				while((dx.dot(dx)) > 0.25*r_cutoff2)
				{
					message_step_limitation = 1;
					dx = 0.1*dx;
				}

				// apply increment and compute new iteration values
				x = x + dx(Eigen::seq(0, dim - 1));
				lambda = lambda + dx[dim];

				// prepare values for next iteration and update the exit criterion
				xax = x - xa;
				if (!minterpol)
				{
                    			p = get_p(x, c, polynomialDegree);
                    			grad_p = get_grad_p(x, c, polynomialDegree);
                		}
				if (minterpol)
                		{
                    			p = get_p_minter(x, model);
				    	grad_p = get_grad_p_minter(x, model);
                		}
				nabla_f(Eigen::seq(0, dim - 1)) = xax + lambda*grad_p;
				nabla_f[dim] = p;

				//norm_dx = dx.norm(); // alternative criterion: incremental tolerance
				nabla_f_norm = nabla_f.norm();

				++k_newton;

				if(redistOptions.verbose)
				{
					std::cout<<"dx: "<<dx[0]<<", "<<dx[1]<<std::endl;
					std::cout<<"H_f:\n"<<H_f<<"\nH_f_inv:\n"<<H_f.inverse()<<std::endl;
					std::cout<<"x:"<<x[0]<<", "<<x[1]<<"\nc:"<<std::endl;
					std::cout<<c<<std::endl;
					std::cout<<"dpdx: "<<grad_p<<std::endl;
					std::cout<<"k = "<<k_newton<<std::endl;
					std::cout<<"x_k = "<<x[0]<<", "<<x[1]<<std::endl;
					std::cout<<x[0]<<", "<<x[1]<<", "<<nabla_f_norm<<std::endl;
				}
			}

			// Check if the Newton algorithm achieved the required accuracy within the allowed number of iterations.
			// If not, throw a warning, but proceed as usual (even if the accuracy dictated by the user cannot be reached,
			// the result can still be very good.
			if (k_newton == redistOptions.max_iter) message_convergence_problem = 1;

			// And finally, compute the new sdf value as the distance of the query particle x_a and the result of the
			// optimization scheme x. We conserve the initial sign of the sdf, since the particle moves with the phase
			// and does not cross the interface.
			if (redistOptions.write_sdf) vd_in.template getProp<vd_in_sdf>(a) = return_sign(vd_in.template getProp<vd_in_sdf>(a))*xax.norm();
			if (redistOptions.write_cp) for(int k = 0; k <dim; k++) vd_in.template getProp<vd_in_cp>(a)[k] = x[k];
	    		// This is to avoid loss of mass conservation - in some simulations, a particle can get assigned a 0-sdf, so
	   		// that it lies on the surface and cannot be distinguished from the one phase or the other. The reason for
	    		// this probably lies in the numerical tolerance. As a quick fix, we introduce an error of the tolerance,
	    		// but keep the sign.
            		if ((k_newton == 0) && (xax.norm() < redistOptions.tolerance))
            		{
                		vd_in.template getProp<vd_in_sdf>(a) = return_sign(vd_in.template getProp<vd_in_sdf>(a))*redistOptions.tolerance;
            		}

            		// debug optimisation
			if(redistOptions.verbose)
			{
				//std::cout<<"new sdf: "<<vd.getProp<sdf>(a)<<std::endl;
				//std::cout<<"c:\n"<<vd.getProp<interpol_coeff>(a)<<"\nmin_sdf: "<<vd.getProp<min_sdf>(a)<<" , min_sdf_x: "<<vd.getProp<min_sdf_x>(a)<<std::endl;
				std::cout<<"x_final: "<<x[0]<<", "<<x[1]<<std::endl;
				if (!minterpol)
                		{
					std::cout<<"p(x_final) :"<<get_p(x, c, polynomialDegree)<<std::endl;
					std::cout<<"nabla p(x_final)"<<get_grad_p(x, c, polynomialDegree)<<std::endl;
                		}
				else
                		{
                    			std::cout<<"p(x_final) :"<<get_p_minter(x, model)<<std::endl;
                    			std::cout<<"nabla p(x_final)"<<get_grad_p_minter(x, model)<<std::endl;
                		}
					std::cout<<"lambda: "<<lambda<<std::endl;
			}

			if (redistOptions.compute_normals)
			{
				for(int k = 0; k<dim; k++) vd_in.template getProp<vd_in_normal>(a)[k] = return_sign(vd_in.template getProp<vd_in_sdf>(a))*grad_p(k)*1/grad_p.norm();
			}

			if (redistOptions.compute_curvatures)
			{
				if (!minterpol) H_p = get_H_p(x, c, polynomialDegree);
				if (minterpol) H_p = get_H_p_minter(x, model);
				// divergence of normalized gradient field:
				if (dim == 2)
				{
					vd_in.template getProp<vd_in_curvature>(a) = (H_p(0,0)*grad_p(1)*grad_p(1) - 2*grad_p(1)*grad_p(0)*H_p(0,1) + H_p(1,1)*grad_p(0)*grad_p(0))/std::pow(sqrt(grad_p(0)*grad_p(0) + grad_p(1)*grad_p(1)),3);
				}
				else if (dim == 3)
				{	// Mean curvature is 0.5*fluid mechanical curvature (see https://link.springer.com/article/10.1007/s00466-021-02128-9)
					vd_in.template getProp<vd_in_curvature>(a) = 0.5*
					((H_p(1,1) + H_p(2,2))*std::pow(grad_p(0), 2) + (H_p(0,0) + H_p(2,2))*std::pow(grad_p(1), 2) + (H_p(0,0) + H_p(1,1))*std::pow(grad_p(2), 2)
					- 2*grad_p(0)*grad_p(1)*H_p(0,1) - 2*grad_p(0)*grad_p(2)*H_p(0,2) - 2*grad_p(1)*grad_p(2)*H_p(1,2))*std::pow(std::pow(grad_p(0), 2) + std::pow(grad_p(1), 2) + std::pow(grad_p(2), 2), -1.5);
				}

			}
			++part;
		}
		if (message_step_limitation)
		{
			std::cout<<"Step size limitation invoked"<<std::endl;
		}
		if (message_convergence_problem)
		{
			std::cout<<"Warning: Newton algorithm has reached maximum number of iterations, does not converge for some particles"<<std::endl;
		}

	}

	// monomial regression polynomial evaluation
	inline double get_p(EMatrix<double, Eigen::Dynamic, 1> xvector, EMatrix<double, Eigen::Dynamic, 1> c, std::string polynomialdegree)
	{
		const double x = xvector[0];
		const double y = xvector[1];

		if (dim == 2)
		{
			if (polynomialdegree == "bicubic")
			{
				return(c[0] + c[1]*x + c[2]*x*x + c[3]*x*x*x + c[4]*y + c[5]*x*y + c[6]*x*x*y + c[7]*x*x*x*y + c[8]*y*y + c[9]*x*y*y + c[10]*x*x*y*y + c[11]*x*x*x*y*y +c[12]*y*y*y + c[13]*x*y*y*y + c[14]*x*x*y*y*y + c[15]*x*x*x*y*y*y);
			}
			if (polynomialdegree == "quadratic")
			{
				return(c[0] + c[1]*x + c[2]*x*x + c[3]*y + c[4]*x*y + c[5]*y*y);
			}
			if (polynomialdegree == "taylor4")
			{
				return(c[0] + c[1]*x + c[2]*x*x + c[3]*x*x*x + c[4]*x*x*x*x + c[5]*y + c[6]*x*y + c[7]*x*x*y + c[8]*x*x*x*y + c[9]*y*y + c[10]*y*y*x + c[11]*x*x*y*y + c[12]*y*y*y + c[13]*y*y*y*x + c[14]*y*y*y*y);
			}
			else throw std::invalid_argument("received unknown polynomial degree");
		}
		if (dim == 3)
		{
			const double z = xvector[2];
			if (polynomialdegree == "taylor4")
			{
				return(c[0] + c[1]*x + c[2]*x*x + c[3]*x*x*x + c[4]*x*x*x*x + c[5]*y + c[6]*x*y + c[7]*x*x*y + c[8]*x*x*x*y + c[9]*y*y + c[10]*y*y*x + c[11]*x*x*y*y + c[12]*y*y*y + c[13]*y*y*y*x + c[14]*y*y*y*y
							+ c[15]*z + c[16]*x*z + c[17]*x*x*z + c[18]*x*x*x*z + c[19]*y*z + c[20]*x*y*z + c[21]*x*x*y*z + c[22]*y*y*z + c[23]*x*y*y*z + c[24]*y*y*y*z + c[25]*z*z + c[26]*x*z*z + c[27]*x*x*z*z
							+ c[28]*y*z*z + c[29]*x*y*z*z + c[30]*y*y*z*z + c[31]*z*z*z + c[32]*x*z*z*z + c[33]*y*z*z*z + c[34]*z*z*z*z);
			}
			else throw std::invalid_argument("received unknown polynomial degree");
		}
		else throw std::invalid_argument("received unknown input dimension. Spatial variable needs to be either 2D or 3D.");
	}

	inline EMatrix<double, Eigen::Dynamic, 1> get_grad_p(EMatrix<double, Eigen::Dynamic, 1> xvector, EMatrix<double, Eigen::Dynamic, 1> c, std::string polynomialdegree)
	{
		const double x = xvector[0];
		const double y = xvector[1];
		EMatrix<double, Eigen::Dynamic, 1> grad_p(dim_r, 1);

		if (dim == 2)
		{
			if (polynomialdegree == "quadratic")
			{
				grad_p[0] = c[1] + 2*c[2]*x + c[4]*y;
				grad_p[1] = c[3] + c[4]*x + 2*c[5]*y;
			}
			else if (polynomialdegree == "bicubic")
			{
				grad_p[0] = c[1] + 2*c[2]*x + 3*c[3]*x*x + c[5]*y + 2*c[6]*x*y + 3*c[7]*x*x*y + c[9]*y*y + 2*c[10]*x*y*y + 3*c[11]*x*x*y*y + c[13]*y*y*y + 2*c[14]*x*y*y*y + 3*c[15]*x*x*y*y*y;
				grad_p[1] = c[4] + c[5]*x + c[6]*x*x + c[7]*x*x*x + 2*c[8]*y + 2*c[9]*x*y + 2*c[10]*x*x*y + 2*c[11]*x*x*x*y + 3*c[12]*y*y + 3*c[13]*x*y*y + 3*c[14]*x*x*y*y + 3*c[15]*x*x*x*y*y;
			}
			else if (polynomialdegree == "taylor4")
			{
				grad_p[0] = c[1] + 2*c[2]*x + 3*c[3]*x*x + 4*c[4]*x*x*x + c[6]*y + 2*c[7]*x*y + 3*c[8]*x*x*y + c[10]*y*y + 2*c[11]*x*y*y + c[13]*y*y*y;
				grad_p[1] = c[5] + c[6]*x + c[7]*x*x + c[8]*x*x*x + 2*c[9]*y + 2*c[10]*x*y + 2*c[11]*x*x*y + 3*c[12]*y*y + 3*c[13]*y*y*x + 4*c[14]*y*y*y;
			}
			else throw std::invalid_argument("received unknown polynomial degree");
		}
		else if (dim == 3)
		{
			const double z = xvector[2];
			if (polynomialdegree == "taylor4")
			{
				grad_p[0] = c[1] + 2*c[2]*x + 3*c[3]*x*x + 4*c[4]*x*x*x + c[6]*y + 2*c[7]*x*y + 3*c[8]*x*x*y + c[10]*y*y + 2*c[11]*x*y*y + c[13]*y*y*y
								+ c[16]*z + 2*c[17]*x*z + 3*c[18]*x*x*z + c[20]*y*z + 2*c[21]*x*y*z + c[23]*y*y*z + c[26]*z*z + 2*c[27]*x*z*z + c[29]*y*z*z + c[32]*z*z*z;
				grad_p[1] = c[5] + c[6]*x + c[7]*x*x + c[8]*x*x*x + 2*c[9]*y + 2*c[10]*x*y + 2*c[11]*x*x*y + 3*c[12]*y*y + 3*c[13]*y*y*x + 4*c[14]*y*y*y
								+ c[19]*z + c[20]*x*z + c[21]*x*x*z + 2*c[22]*y*z + 2*c[23]*x*y*z + 3*c[24]*y*y*z + c[28]*z*z + c[29]*x*z*z + 2*c[30]*y*z*z + c[33]*z*z*z;
				grad_p[2] = c[15] + c[16]*x + c[17]*x*x + c[18]*x*x*x + c[19]*y + c[20]*x*y + c[21]*x*x*y + c[22]*y*y + c[23]*x*y*y + c[24]*y*y*y
								+ 2*c[25]*z + 2*c[26]*x*z + 2*c[27]*x*x*z + 2*c[28]*y*z + 2*c[29]*x*y*z + 2*c[30]*y*y*z + 3*c[31]*z*z + 3*c[32]*x*z*z + 3*c[33]*y*z*z + 4*c[34]*z*z*z;
			}
		}
		else throw std::invalid_argument("received unknown input dimension. Spatial variable needs to be either 2D or 3D.");

		return(grad_p);

	}

	inline EMatrix<double, Eigen::Dynamic, Eigen::Dynamic> get_H_p(EMatrix<double, Eigen::Dynamic, 1> xvector, EMatrix<double, Eigen::Dynamic, 1> c, std::string polynomialdegree)
	{
		const double x = xvector[0];
		const double y = xvector[1];
		EMatrix<double, Eigen::Dynamic, Eigen::Dynamic> H_p(dim_r, dim_r);

		if (dim == 2)
		{
			if (polynomialdegree == "quadratic")
			{
				H_p(0, 0) = 2*c[2];
				H_p(0, 1) = c[4];
				H_p(1, 0) = H_p(0, 1);
				H_p(1, 1) = 2*c[5];
			}
			else if (polynomialdegree == "bicubic")
			{
				H_p(0, 0) = 2*c[2] + 6*c[3]*x + 2*c[6]*y + 6*c[7]*x*y + 2*c[10]*y*y + 6*c[11]*y*y*x + 2*c[14]*y*y*y + 6*c[15]*y*y*y*x;
				H_p(0, 1) = c[5] + 2*c[6]*x + 3*c[7]*x*x + 2*c[9]*y + 4*c[10]*x*y + 6*c[11]*x*x*y + 3*c[13]*y*y + 6*c[14]*x*y*y + 9*c[15]*x*x*y*y;
				H_p(1, 0) = H_p(0, 1);
				H_p(1, 1) = 2*c[8] + 2*c[9]*x + 2*c[10]*x*x + 2*c[11]*x*x*x + 6*c[12]*y + 6*c[13]*x*y + 6*c[14]*x*x*y + 6*c[15]*x*x*x*y;
			}
			else if (polynomialdegree == "taylor4")
			{
				H_p(0, 0) = 2*c[2] + 6*c[3]*x + 12*c[4]*x*x + 2*c[7]*y + 6*c[8]*x*y + 2*c[11]*y*y;
				H_p(0, 1) = c[6] + 2*c[7]*x + 3*c[8]*x*x + 2*c[10]*y +4*c[11]*x*y + 3*c[13]*y*y;
				H_p(1, 0) = H_p(0, 1);
				H_p(1, 1) = 2*c[9] + 2*c[10]*x + 2*c[11]*x*x + 6*c[12]*y + 6*c[13]*x*y + 12*c[14]*y*y;
			}
			else throw std::invalid_argument("received unknown polynomial degree");
		}
		else if (dim == 3)
		{
			const double z = xvector[2];
			if (polynomialdegree == "taylor4")
			{
				H_p(0, 0) = 2*c[2] + 6*c[3]*x + 12*c[4]*x*x + 2*c[7]*y + 6*c[8]*x*y + 2*c[11]*y*y
							+ 2*c[17]*z + 6*c[18]*x*z + 2*c[21]*y*z + 2*c[27]*z*z;
				H_p(1, 1) = 2*c[9] + 2*c[10]*x + 2*c[11]*x*x + 6*c[12]*y + 6*c[13]*x*y + 12*c[14]*y*y
							+ 2*c[22]*z + 2*c[23]*x*z + 6*c[24]*y*z + 2*c[30]*z*z;
				H_p(2, 2) = 2*c[25] + 2*c[26]*x + 2*c[27]*x*x + 2*c[28]*y + 2*c[29]*x*y + 2*c[30]*y*y
							+ 6*c[31]*z + 6*c[32]*x*z + 6*c[33]*y*z + 12*c[34]*z*z;

				H_p(0, 1) = (c[6] + 2*c[7]*x + 3*c[8]*x*x + 2*c[10]*y +4*c[11]*x*y + 3*c[13]*y*y
							+ c[20]*z + 2*c[21]*x*z + 2*c[23]*y*z + c[29]*z*z);
				H_p(0, 2) = (c[16] + 2*c[17]*x + 3*c[18]*x*x + c[20]*y + 2*c[21]*x*y + c[23]*y*y
							+ 2*c[26]*z + 4*c[27]*x*z + 2*c[29]*y*z + 3*c[32]*z*z);
				H_p(1, 2) = (c[19] + c[20]*x + c[21]*x*x + 2*c[22]*y + 2*c[23]*x*y + 3*c[24]*y*y
							+ 2*c[28]*z + 2*c[29]*x*z + 4*c[30]*y*z + 3*c[33]*z*z);
				H_p(1, 0) = H_p(0, 1);
				H_p(2, 0) = H_p(0, 2);
				H_p(2, 1) = H_p(1, 2);
			}
		}
		else throw std::invalid_argument("received unknown input dimension. Spatial variable needs to be either 2D or 3D.");

		return(H_p);

	}

	// minterface
	template<typename PolyType>
	inline double get_p_minter(EMatrix<double, Eigen::Dynamic, 1> xvector, PolyType model)
    	{
        	return(model->eval(xvector.transpose())(0));
    	}


	template<typename PolyType>
	inline EMatrix<double, Eigen::Dynamic, 1> get_grad_p_minter(EMatrix<double, Eigen::Dynamic, 1> xvector, PolyType model)
    	{
        	EMatrix<double, Eigen::Dynamic, 1> grad_p(dim, 1);
        	std::vector<int> derivOrder(dim, 0);
       		for(int k = 0; k < dim; k++){
            		std::fill(derivOrder.begin(), derivOrder.end(), 0);
            		derivOrder[k] = 1;
            		grad_p[k] = model->deriv_eval(xvector.transpose(), derivOrder)(0);
        	}
        	return(grad_p);
    	}

	template<typename PolyType>
    	inline EMatrix<double, Eigen::Dynamic, Eigen::Dynamic> get_H_p_minter(EMatrix<double, Eigen::Dynamic, 1> xvector, PolyType model)
    	{
        	EMatrix<double, Eigen::Dynamic, Eigen::Dynamic> H_p(dim, dim);
       		std::vector<int> derivOrder(dim, 0);

        	for(int k = 0; k < dim; k++){
            		for(int l = 0; l < dim; l++)
            		{
                		std::fill(derivOrder.begin(), derivOrder.end(), 0);
                		derivOrder[k]++;
                		derivOrder[l]++;
                		H_p(k, l) = model->deriv_eval(xvector.transpose(), derivOrder)(0);
            		}
        	}
        	return(H_p);
    	}

};


