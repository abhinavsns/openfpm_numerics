//
// Created by jstark on 2020-05-16.
//
/**
 * @file NarrowBand.hpp
 * @class NarrowBand
 *
 * @brief Class for getting the narrow band around the interface.
 *
 * @details Places particle at those grid point positions, which lay inside a narrow band of user-defined width
 * around the interface. The respective SDF or arbitrary other property value is then also copied from the grid node
 * to that particle occupying the same position.
 *
 *
 * @author Justina Stark
 * @date May 2020
 */

#ifndef REDISTANCING_SUSSMAN_NARROWBAND_HPP
#define REDISTANCING_SUSSMAN_NARROWBAND_HPP

// Include standard library header files
#include <iostream>

// Include OpenFPM header files
#include "Vector/vector_dist.hpp"
#include "Grid/grid_dist_id.hpp"
#include "data_type/aggregate.hpp"
#include "Decomposition/CartDecomposition.hpp"

// Include level-set-method related header files
#include "HelpFunctions.hpp"
#include "HelpFunctionsForGrid.hpp"
#include "FiniteDifference/Upwind_gradient.hpp"

/**@brief Class for getting the narrow band around the interface
 * @file NarrowBand.hpp
 * @class NarrowBand
 * @tparam grid_in_type Template type of input grid that stores the signed distance function Phi_SDF.
 */
template <typename grid_in_type, typename phi_type=double>
class NarrowBand
{
public:
	/** @brief Constructor taking the thickness of the narrow band as #grid points in order to initialize the
	 * lower and upper bound for the narrow band. Initializes the temporary internal grid.
	 *
	 * @param grid_in Input grid with min. 1 property: Phi_SDF.
	 * @param thickness Width of narrow band in # grid points.
	 */
	NarrowBand(const grid_in_type & grid_in,
	           size_t thickness) // thickness in # grid points
			: g_temp(grid_in.getDecomposition(), grid_in.getGridInfoVoid().getSize(), Ghost<grid_in_type::dims, long int>(3))
	{
		set_bounds(thickness, grid_in);
	}
	/** @brief Constructor taking the thickness of the narrow band as physical width in space in order to initialize the
	 * lower and upper bound for the narrow band. Initializes the temporary internal grid.
	 *
	 * @param grid_in Input grid with min. 1 property: Phi_SDF.
	 * @param thickness Width of narrow band as physical width.
	 */
	NarrowBand(const grid_in_type & grid_in,
	           double thickness)    // thickness as physical width
			: g_temp(grid_in.getDecomposition(), grid_in.getGridInfoVoid().getSize(), Ghost<grid_in_type::dims, long int>(3))
	{
		set_bounds(thickness, grid_in);
	}
	/** @brief Constructor taking the thickness of the narrow band as physical width in space in order to initialize the
     * lower and upper bound for the narrow band. Initializes the temporary internal grid.
	 *
     * @param grid_in Input grid with min. 1 property: Phi_SDF.
     * @param thickness Width of narrow band as physical width.
     */
	NarrowBand(const grid_in_type & grid_in,
	           float thickness)    // thickness as physical width
			: g_temp(grid_in.getDecomposition(), grid_in.getGridInfoVoid().getSize(), Ghost<grid_in_type::dims, long int>(3))
	{
		set_bounds(thickness, grid_in);
	}
	/** @brief Constructor taking the inside and outside physical width of the narrow band in order to initialize the
	 * lower and upper bound for the narrow band. Initializes the temporary internal grid.
	 *
	 * @param grid_in Input grid with min. 1 property: Phi_SDF.
	 * @param width_outside Extension of narrow band as physical width inside of object (Phi > 0).
	 * @param width_inside Extension of narrow band as physical width outside of object (Phi < 0).
	 */
	template<typename width_type>
	NarrowBand(const grid_in_type & grid_in,
	           width_type width_outside,     // physical width of nb inside of object -> Phi > 0
	           width_type width_inside)     // physical width nb outside of object -> Phi < 0
			: g_temp(grid_in.getDecomposition(), grid_in.getGridInfoVoid().getSize(), Ghost<grid_in_type::dims, long int>(3))
	{
		set_bounds(width_outside, width_inside, grid_in);
	}
	/**@brief Aggregated properties for the temporary grid.
	 *
	 * @details The properties are Phi_SDF (result from redistancing), gradient of Phi, and sign of Phi_SDF (for the upwinding).
	 * */
	typedef aggregate<phi_type, phi_type[grid_in_type::dims], int> props_temp;
	/** @brief Type definition for the temporary grid.
	 */
	typedef grid_dist_id<grid_in_type::dims, typename grid_in_type::stype, props_temp> g_temp_type;
	/**
	 * @brief Create temporary grid, which is only used inside the class to get the gradients.
	 *
	 * @details The grid is needed, when the user wants a narrow band which also contains the upwind gradient of phi.
	 * It contains the following 4 properties: Phi_SDF (result from redistancing), gradient of Phi, and sign of
	 * Phi_SDF (for the upwinding).
	 * */
	g_temp_type g_temp;
	
	/** @brief Places particles within a narrow band around the interface.
	 * Only the SDF is copied from the grid properties to the respective particles.
	 *
	 * @tparam Phi_SDF_grid Index of property storing the signed distance function in the input grid.
	 * @tparam Phi_SDF_vd Index of property that should store the SDF in the narrow band particle vector.
	 * @tparam vector_type Template type of the particle vector.
	 * @tparam grid_type Template type of the grid storing the SDF.
	 *
	 * @param grid Grid of arb. dims. storing the SDF (result of redistancing).
	 * @param vd Empty vector with same spatial scaling (box) as the grid.
	 */
	template <size_t Phi_SDF_grid, size_t Phi_SDF_vd, typename vector_type, typename grid_type>
	void get_narrow_band(grid_type & grid, vector_type & vd)
	{
		auto dom = grid.getDomainIterator();
		while(dom.isNext())
		{
			auto key = dom.get();
			if(within_narrow_band(grid.template get<Phi_SDF_grid>(key)))
			{
				// add particle to vd_narrow_band
				vd.add();
				// assign coordinates and properties from respective grid point to particle
				for(size_t d = 0; d < grid_type::dims; d++)
				{
					vd.getLastPos()[d] = grid.getPos(key)[d];
				}
				vd.template getLastProp<Phi_SDF_vd>() = grid.template get<Phi_SDF_grid>(key);
			}
			++dom;
		}
		vd.map();
	}
	/** @brief Places particles within a narrow band around the interface.
	 * SDF and Phi_grad_temp are copied from the temp. grid to the respective particle.
	 *
	 * @tparam Phi_SDF_grid Index of property storing the signed distance function in the input grid.
	 * @tparam Phi_SDF_vd Index of property that should store the SDF in the narrow band particle vector.
	 * @tparam Phi_grad Index of property that should store the gradient of phi in the narrow band particle vector.
	 * @tparam vector_type Template type of the particle vector.
	 * @tparam grid_type Template type of the grid storing the SDF.
	 *
	 * @param grid Grid of arb. dims. storing the SDF (result of redistancing).
	 * @param vd Empty vector with same spatial scaling (box) as the grid.
	 */
	template <size_t Phi_SDF_grid, size_t Phi_SDF_vd, size_t Phi_grad, typename vector_type, typename grid_type>
	void get_narrow_band(grid_type & grid, vector_type & vd)
	{
		initialize_temporary_grid<Phi_SDF_grid>(grid);
		auto dom = grid.getDomainIterator();
		while(dom.isNext())
		{
			auto key = dom.get();
			if(within_narrow_band(grid.template get<Phi_SDF_temp>(key)))
			{
				// add particle to vd_narrow_band
				vd.add();
				// assign coordinates and properties from respective grid point to particle
				for(size_t d = 0; d < grid_type::dims; d++)
				{
					vd.getLastPos()[d] = grid.getPos(key)[d];
					vd.template getLastProp<Phi_grad>()[d] = g_temp.template get<Phi_grad_temp>(key)[d];
				}
				vd.template getLastProp<Phi_SDF_vd>() = grid.template get<Phi_SDF_temp>(key);
			}
			++dom;
		}
		vd.map();
	}
	/** @brief Places particles within a narrow band around the interface.
	 * SDF and Phi_grad_temp are copied from the temp. grid to the respective particle.
	 *
	 * @tparam Phi_SDF_grid Index of property storing the signed distance function in the input grid.
	 * @tparam Phi_SDF_vd Index of property that should store the SDF in the narrow band particle vector.
	 * @tparam Phi_grad Index of property that should store the gradient of phi in the narrow band particle vector.
	 * @tparam Phi_magnOfGrad Index of property that should store the gradient magnitude of Phi.
	 * @tparam vector_type Template type of the particle vector.
	 * @tparam grid_type Template type of the grid storing the SDF.
	 *
	 * @param grid Grid of arb. dims. storing the SDF (result of redistancing).
	 * @param vd Empty vector with same spatial scaling (box) as the grid.
	 */
	template <size_t Phi_SDF_grid, size_t Phi_SDF_vd, size_t Phi_grad, size_t Phi_magnOfGrad, typename vector_type,
			typename grid_type>
	void get_narrow_band(grid_type & grid, vector_type & vd)
	{
		initialize_temporary_grid<Phi_SDF_grid>(grid);
		auto dom = grid.getDomainIterator();
		while(dom.isNext())
		{
			auto key = dom.get();
			if(within_narrow_band(grid.template get<Phi_SDF_grid>(key)))
			{
				// add particle to vd_narrow_band
				vd.add();
				// assign coordinates and properties from respective grid point to particle
				for(size_t d = 0; d < grid_type::dims; d++)
				{
					vd.getLastPos()[d] = grid.getPos(key)[d];
					vd.template getLastProp<Phi_grad>()[d] = g_temp.template get<Phi_grad_temp>(key)[d];
				}
				vd.template getLastProp<Phi_SDF_vd>()      = g_temp.template get<Phi_SDF_temp>(key);
				vd.template getLastProp<Phi_magnOfGrad>()  = get_vector_magnitude<Phi_grad_temp>(g_temp, key);
			}
			++dom;
		}
		vd.map();	}
	/** @brief Places particles within a narrow band around the interface. An arbitrary property is copied from the
	 * grid to the respective particle.
	 *
	 * @tparam Phi_SDF_grid Index of property storing the signed distance function in the input grid.
	 * @tparam Prop1_grid Index of arbitrary grid property that should be copied to the particles (prop. value must
	 * be a scalar).
	 * @tparam Prop1_vd Index of particle property, where grid property should be copied to (prop. value must be a
	 * scalar).
	 * @tparam vector_type Template type of the particle vector.
	 * @tparam grid_type Template type of the grid storing the SDF.
	 *
	 * @param grid Grid of arb. dims. storing the SDF (result of redistancing) and any arbitrary other (scalar)
	 * property.
	 * @param vd Empty vector with same spatial scaling (box) as the grid.
	 */
	template <size_t Phi_SDF_grid, size_t Prop1_grid, size_t Prop1_vd, typename vector_type, typename grid_type>
	void get_narrow_band_copy_specific_property(grid_type & grid, vector_type & vd)
	{
		auto dom = grid.getDomainIterator();
		while(dom.isNext())
		{
			auto key = dom.get();
			if(within_narrow_band(grid.template get<Phi_SDF_grid>(key)))
			{
				// add particle to vd_narrow_band
				vd.add();
				// assign coordinates and properties from respective grid point to particle
				for(size_t d = 0; d < grid_type::dims; d++)
				{
					vd.getLastPos()[d] = grid.getPos(key)[d];
				}
				vd.template getLastProp<Prop1_vd>() = grid.template get<Prop1_grid>(key);
			}
			++dom;
		}
		vd.map();
	}
	/**@brief Places particles within a narrow band around the interface. An arbitrary property is copied from the
	 * grid to the respective particle.
	 * @tparam Phi_SDF_grid Index of property storing the signed distance function in the input grid.
	 * @tparam Index1Grid Index of 1st scalar grid property that should be copied from grid to particles.
	 * @tparam Index2Grid Index of 2nd scalar grid property that should be copied from grid to particles.
	 * @tparam Index3Grid Index of 3rd scalar grid property that should be copied from grid to particles.
	 * @tparam Index1Vd Index of 1st scalar property on particle to which value from grid should be copied to.
	 * @tparam Index2Vd Index of 2nd scalar property on particle to which value from grid should be copied to.
	 * @tparam Index3Vd Index of 3rd scalar property on particle to which value from grid should be copied to.
	 * @tparam vector_type Template type of the particle vector.
	 * @tparam grid_type Template type of the grid storing the SDF.
	 *
	 * @param grid Grid of arb. dims. storing the SDF (result of redistancing) and any arbitrary other (scalar)
	 * property.
	 * @param vd Empty vector with same spatial scaling (box) as the grid.
	 */
	template <size_t Phi_SDF_grid, size_t Index1Grid, size_t Index2Grid, size_t Index3Grid, size_t Index1Vd, size_t
	Index2Vd, size_t Index3Vd, typename grid_type, typename vector_type>
	void get_narrow_band_copy_three_scalar_properties(grid_type & grid, vector_type & vd)
	{
		
			auto dom = grid.getDomainIterator();
			while(dom.isNext())
			{
				auto key = dom.get();
				auto key_g = grid.getGKey(key);
				if(within_narrow_band(grid.template get<Phi_SDF_grid>(key)))
				{
					// add particle to vd_narrow_band
					vd.add();
					// assign coordinates and properties from respective grid point to particle
					for(size_t d = 0; d < grid_type::dims; d++)
					{
						vd.getLastPos()[d] = key_g.get(d) * grid.getSpacing()[d];
					}
					vd.template getLastProp<Index1Vd>() = grid.template get<Index1Grid>(key);
					vd.template getLastProp<Index2Vd>() = grid.template get<Index2Grid>(key);
					vd.template getLastProp<Index3Vd>() = grid.template get<Index3Grid>(key);
				}
				++dom;
			}
			vd.map();
	}
private:
	//	Some indices for better readability
	static const size_t Phi_SDF_temp        = 0; ///< Property index of Phi_SDF on the temporary grid.
	static const size_t Phi_grad_temp       = 1; ///< Property index of gradient of Phi on the temporary grid.
	static const size_t Phi_sign_temp       = 2; ///< Property index of sign of Phi on the temporary grid.
	
	phi_type b_low;   ///< Narrow band extension towards the object outside.
	phi_type b_up;    ///< Narrow band extension towards the object inside.
	
	/** @brief Set the member variable NarrowBand::b_low and NarrowBand::b_up.
	 *
	 * @param thickness Thickness of narrow band in number of grid points.
	 * @param grid_in Input grid storing the signed distance function Phi_SDF (redistancing output).
	 */
	void set_bounds(const size_t thickness, const grid_in_type & grid_in)
	{
		const phi_type EPSILON = std::numeric_limits<phi_type>::epsilon();
		b_low   = - ceil((phi_type)thickness / 2.0) * get_biggest_spacing(grid_in) - EPSILON;
		b_up    =   ceil((phi_type)thickness / 2.0) * get_biggest_spacing(grid_in) + EPSILON;
	}
	/** @brief Set the member variable NarrowBand::b_low and NarrowBand::b_up.
	 *
	 * @param thickness Thickness of narrow band as physical width.
	 * @param grid_in Input grid storing the signed distance function Phi_SDF (redistancing output).
	 */
	template<typename thickness_type>
	void set_bounds(const thickness_type thickness, const grid_in_type & grid_in)
	{
		const thickness_type EPSILON = std::numeric_limits<thickness_type>::epsilon();
		b_low   = - thickness / 2.0 - EPSILON;
		b_up    =   thickness / 2.0 + EPSILON;
	}
	/** @brief Set the member variable NarrowBand::b_low and NarrowBand::b_up.
	 *
	 * @param lower_bound Extension of narrow band as physical width inside of object (Phi > 0).
	 * @param upper_bound Extension of narrow band as physical width outside of object (Phi < 0).
	 * @param grid_in Input grid storing the signed distance function Phi_SDF (redistancing output).
	 */
	template<typename width_type>
	void set_bounds(width_type lower_bound, width_type upper_bound, const grid_in_type & grid_in)
	{
		const width_type EPSILON = std::numeric_limits<width_type>::epsilon();
		b_low   =   lower_bound - EPSILON;
		b_up    =   upper_bound + EPSILON;
	}
	
	/**@brief Initialize the internal temporary grid.
	 *
	 * @details Copies Phi_SDF from the input grid to the temorary grid.
	 *
	 * @tparam Phi_SDF Index of property storing the signed distance function in the input grid.
	 * @param grid_in Input grid storing the signed distance function Phi_SDF (redistancing output).
	 */
	template <size_t Phi_SDF>
	void initialize_temporary_grid(const grid_in_type & grid_in)
	{
		copy_gridTogrid<Phi_SDF, Phi_SDF_temp>(grid_in, g_temp); // Copy Phi_SDF from the input grid to the temorary grid
		init_sign_prop<Phi_SDF_temp, Phi_sign_temp>(g_temp); // initialize Phi_sign_temp with the sign of the
		// input Phi_SDF
		get_upwind_gradient<Phi_SDF_temp, Phi_sign_temp, Phi_grad_temp>(g_temp, 1, true);   // Get initial gradients
	}
	/**@brief Checks if a value for Phi_SDF lays within the narrow band.
	 *
	 * @tparam T Template type of parameter Phi.
	 * @param Phi Value of the signed distance function Phi_SDF.
	 * @return True, if point lays within the narrow band; False if not.
	 */
	template<typename T>
	bool within_narrow_band(T Phi)
	{
		return (Phi >= b_low && Phi <= b_up);
	}
};


#endif //REDISTANCING_SUSSMAN_NARROWBAND_HPP
