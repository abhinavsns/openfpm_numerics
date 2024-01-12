/**
 * @file closest_point.hpp
 *
 *
 * @brief Functions for level set reinitialization and extension on OpenFPM grids based
 *        on closest point method.
 *
 * @details Depends on Algoim library for higher order closest point calculations and constructing
 *          stencil interpolating polynomials.
 *
 *
 * @author Sachin Krishnan T V
 * @date May 2021
 */


#ifndef __CLOSEST_POINT_HPP__
#define __CLOSEST_POINT_HPP__

#include "Grid/grid_dist_key.hpp"
#include "algoim_hocp.hpp"

// Width of extra padding around each grid patch needed to correctly construct kDTree in Algoim.
constexpr int algoim_padding = 4;

/**@brief Wrapping container to pass OpenFPM grid property values to Algoim library.
 *
 * @file closest_point.hpp
 * @struct AlgoimWrapper
 * @tparam wrapping_field Property id on the grid for the field to be wrapped
 * @tparam grid_type Type of the grid container
 * 
 */

template<size_t wrapping_field, typename grid_type>
struct AlgoimWrapper
{
    const static unsigned int dim = grid_type::dims;
    grid_type &gd;
    int patch_id;
    AlgoimWrapper(grid_type& ls_grid, const int pid) : gd(ls_grid), patch_id(pid) {}

    //! Call operator for the wrapper.
    double operator() (const blitz::TinyVector<int, dim> idx) const
    {
        long int local_key[dim];
        
        auto ghost_offset = gd.getLocalGridsInfo().get(patch_id).Dbox.getKP1();

        for (int d = 0; d < dim; ++d)
            local_key[d] = idx(d) - algoim_padding;

        // Generate OpenFPM grid_key object from local grid indices
        grid_dist_key_dx<dim> grid_key(patch_id, grid_key_dx<dim> (local_key) + ghost_offset);
        
        return gd.template get<wrapping_field>(grid_key);
    }
};

/**@brief Computes the closest point coordinate for each grid point within nb_gamma from interface.
 *
 * @tparam phi_field Property id on grid for the level set SDF (input)
 * @tparam cp_field Property id on grid for storing closest point coordinates (output)
 * @tparam poly_order Type of stencil interpolation (Taylor poly orders between 2 to 5 and Tri/bicubic through -1 is supported)
 * @tparam grid_type Type of the grid container
 *
 * @param gd The distributed grid containing at least level set SDF field and placeholder for closest point coordinates
 * @param nb_gamma The width of the narrow band within which closest point estimation is to be done
 * 
 */
template<size_t phi_field, size_t cp_field, int poly_order, typename grid_type>
void estimateClosestPoint(grid_type &gd, const double nb_gamma)
{
    const unsigned int dim = grid_type::dims;
    // Update the phi field in ghosts
    gd.template ghost_get<phi_field>(KEEP_PROPERTIES);

    // Stencil polynomial type
    using Poly = typename Algoim::StencilPoly<dim, poly_order>::T_Poly;

    // Grid spacing along each dimension
    blitz::TinyVector<double,dim> dx;
    for(int d = 0; d < dim; ++d)
        dx(d) = gd.spacing(d);

    auto &patches = gd.getLocalGridsInfo();

    grid_key_dx<dim> p_lo;
    grid_key_dx<dim> p_hi;

    for(int i = 0; i < patches.size();i++)
    {
        for(int d = 0; d < dim; ++d)
        {
            p_lo.set_d(d, patches.get(i).Dbox.getLow(d) + patches.get(i).origin[d]);
            p_hi.set_d(d, patches.get(i).Dbox.getHigh(d) + patches.get(i).origin[d]);
        }

        AlgoimWrapper<phi_field, grid_type> phiwrap(gd, i);

        // Find all cells containing the interface and construct the high-order polynomials
        std::vector<Algoim::detail::CellPoly<dim,Poly>> cells;

        blitz::TinyVector<int,dim> ext;

        for(int d = 0; d < dim; ++d)
            ext(d) = static_cast<int>(p_hi.get(d) - p_lo.get(d) + 1 + 2*algoim_padding);

        Algoim::detail::createCellPolynomials(ext, phiwrap, dx, false, cells);

        std::vector<blitz::TinyVector<double,dim>> points;
        std::vector<int> pointcells;
        Algoim::detail::samplePolynomials<dim,Poly>(cells, 2, dx, 0.0, points, pointcells);

        Algoim::KDTree<double,dim> kdtree(points);

        // In order to ensure that CP is estimated for all points in the narrowband, we add a buffer to the distance check.
        double nb_gamma_plus_dx = nb_gamma + gd.spacing(0);
        // Pass everything to the closest point computation engine
        Algoim::ComputeHighOrderCP<dim,Poly> hocp(nb_gamma_plus_dx < std::numeric_limits<double>::max() ? nb_gamma_plus_dx*nb_gamma_plus_dx : std::numeric_limits<double>::max(), // squared bandradius
                                        0.5*blitz::max(dx), // amount that each polynomial overlaps / size of the bounding ball in Newton's method
                                        Algoim::sqr(std::max(1.0e-14, std::pow(blitz::max(dx), Poly::order))), // tolerance to determine convergence
                                        cells, kdtree, points, pointcells, dx, 0.0);

        auto it = gd.getSubDomainIterator(p_lo, p_hi);
        while(it.isNext())
        {
            auto key = it.get();
            if(std::abs(gd.template get<phi_field>(key)) < nb_gamma)
            {
                auto key_g = gd.getGKey(key);
                // NOTE: This is not the real grid coordinates, but internal coordinates for algoim
                blitz::TinyVector<double,dim> patch_pos, cp;
                for(int d = 0; d < dim; ++d)
                    patch_pos(d) = (key_g.get(d) - p_lo.get(d) + algoim_padding) * dx(d);

                if (hocp.compute(patch_pos, cp))
                {
                    for(int d = 0; d < dim; ++d)
                        gd.template get<cp_field>(key)[d] = cp(d);
                }
                else
                {
                    std::cout<<"WARN: Closest point computation fails at : ";
                    for(int d = 0; d < dim; ++d)
                    {
                        std::cout<<key_g.get(d)<<" ";
                        gd.template get<cp_field>(key)[d] = -100.0;
                    }
                    std::cout<<"\n";
                }
            }
            ++it;
        }
    }
    return;
}

/**@brief Extends a (scalar) field to within nb_gamma from interface. The grid should have level set SDF and closest point field.
 *
 * @tparam phi_field Property id on grid for the level set SDF
 * @tparam cp_field Property id on grid for storing closest point coordinates
 * @tparam extend_field Property id on grid where the field to be extended resides
 * @tparam extend_field_temp Property id on grid for storing temporary intermediate values
 * @tparam poly_order Type of stencil interpolation (Taylor poly orders between 2 to 5 and Tri/bicubic through -1 is supported)
 * @tparam grid_type Type of the grid container
 *
 * @param gd The distributed grid containing atleast level set SDF field and closest point coordinates
 * @param nb_gamma The width of the narrow band within which extension is required
 */
template<size_t phi_field, size_t cp_field, size_t extend_field, size_t extend_field_temp, int poly_order, typename grid_type>
void extendLSField(grid_type &gd, const double nb_gamma)
{
    const unsigned int dim = grid_type::dims;
    // Update the phi and cp fields in ghost
    gd.template ghost_get<phi_field, cp_field, extend_field>(KEEP_PROPERTIES);

    // Stencil polynomial object
    using Poly = typename Algoim::StencilPoly<dim, poly_order>::T_Poly;
    auto &patches = gd.getLocalGridsInfo();
    blitz::TinyVector<double,dim> dx;
    for(int d = 0; d < dim; ++d)
        dx(d) = gd.spacing(d);

    grid_key_dx<dim> p_lo;
    grid_key_dx<dim> p_hi;

    for(int i = 0; i < patches.size();i++)
    {
        for(int d = 0; d < dim; ++d)
        {
            p_lo.set_d(d, patches.get(i).Dbox.getLow(d) + patches.get(i).origin[d]);
            p_hi.set_d(d, patches.get(i).Dbox.getHigh(d) + patches.get(i).origin[d]);
        }

        auto it = gd.getSubDomainIterator(p_lo, p_hi);

        while(it.isNext())
        {
            auto key = it.get();
            if(std::abs(gd.template get<phi_field>(key)) < nb_gamma)
            {
                blitz::TinyVector<int,dim> coord;
                blitz::TinyVector<double,dim> pos;

                for(int d = 0; d < dim; ++d)
                {
                    double cp_d = gd.template get<cp_field>(key)[d];
                    coord(d) = static_cast<int>(floor(cp_d / gd.spacing(d)));
                    pos(d) = cp_d - coord(d)*gd.spacing(d);
                }

                AlgoimWrapper<extend_field, grid_type> fieldwrap(gd,i);
                Poly field_poly = Poly(coord, fieldwrap, dx);
                // Extension is first done to the temporary field. Otherwise interpolation will be affected.
                gd.template get<extend_field_temp>(key) = field_poly(pos);
            }
            ++it;
        }
    }
    
    // Copy the results to the actual variable
    auto it = gd.getDomainIterator();
    while(it.isNext())
    {
        auto key = it.get();
        if(std::abs(gd.template get<phi_field>(key)) < nb_gamma)
            gd.template get<extend_field>(key) = gd.template get<extend_field_temp>(key);
        ++it;
    }
}

/**@brief Reinitializes the level set Phi field on a grid. The grid should have level set SDF and closest point field.
 *
 * @tparam phi_field Property id on grid for the level set SDF
 * @tparam cp_field Property id on grid for storing closest point coordinates
 * @tparam grid_type Type of the grid container

 * @param gd The distributed grid containing atleast level set SDF field and closest point coordinates
 * @param nb_gamma The width of the narrow band for reinitialization
 */
template<size_t phi_field, size_t cp_field, typename grid_type>
void reinitializeLS(grid_type &gd, const double nb_gamma)
{
    const unsigned int dim = grid_type::dims;

    // Update the cp_field in ghost
    gd.template ghost_get<cp_field>(KEEP_PROPERTIES);

    auto &patches = gd.getLocalGridsInfo();
    blitz::TinyVector<double,dim> dx;
    for(int d = 0; d < dim; ++d)
        dx(d) = gd.spacing(d);

    grid_key_dx<dim> p_lo;
    grid_key_dx<dim> p_hi;

    for(int i = 0; i < patches.size();i++)
    {
        for(int d = 0; d < dim; ++d)
        {
            p_lo.set_d(d, patches.get(i).Dbox.getLow(d) + patches.get(i).origin[d]);
            p_hi.set_d(d, patches.get(i).Dbox.getHigh(d) + patches.get(i).origin[d]);
        }

        auto it = gd.getSubDomainIterator(p_lo, p_hi);

        while(it.isNext())
        {
            auto key = it.get();
            if(std::abs(gd.template get<phi_field>(key)) < nb_gamma)
            {
                // Preserve the current sign of the SDF
                double sign_fn = (gd.template get<phi_field>(key) >= 0.0)?1.0:-1.0;
                auto key_g = gd.getGKey(key);

                // Compute the Euclidean distance from gird coordinate to closest point coordinate
                double distance = 0.0;
                for(int d = 0; d < dim; ++d)
                {
                    // NOTE: This is not the real grid coordinates, but internal coordinates used for algoim
                    double patch_pos = (key_g.get(d) - p_lo.get(d) + algoim_padding) * gd.spacing(d);
                    double cp_d = gd.template get<cp_field>(key)[d];
                    if(cp_d == -100.0)
                        std::cout<<"WARNING: Requesting closest point on nodes where it was not computed."<<std::endl;

                    distance += ((patch_pos - cp_d)*(patch_pos - cp_d));
                }
                distance = sqrt(distance);

                gd.template get<phi_field>(key) = sign_fn*distance;
            }
            ++it;
        }
    }
}

#endif //__CLOSEST_POINT_HPP__
