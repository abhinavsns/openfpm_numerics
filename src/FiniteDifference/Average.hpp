/*
 * Average.hpp
 *
 *  Created on: Nov 18, 2015
 *      Author: Pietro Incardona
 */

#ifndef OPENFPM_NUMERICS_SRC_FINITEDIFFERENCE_AVERAGE_HPP_
#define OPENFPM_NUMERICS_SRC_FINITEDIFFERENCE_AVERAGE_HPP_

#include "FiniteDifference/FD_util_include.hpp"
#include "util/mathutil.hpp"
#include "Vector/map_vector.hpp"
#include "Grid/comb.hpp"
#include "FiniteDifference/util/common.hpp"
#include "util/util_num.hpp"

/*! \brief Average
 *
 * \tparam d on which dimension average
 * \tparam Field which field average
 * \tparam impl which implementation
 *
 */
template<unsigned int d, typename Field, typename Sys_eqs, unsigned int impl=CENTRAL>
class Avg
{
	/*! \brief Create the row of the Matrix
	 *
	 * \tparam ord
	 *
	 */
	inline static void value(const grid_key_dx<Sys_eqs::dims> & pos, const grid_sm<Sys_eqs::dims,void> & gs, std::unordered_map<long int,typename Sys_eqs::stype > & cols, typename Sys_eqs::stype coeff)
	{
		std::cerr << "Error " << __FILE__ << ":" << __LINE__ << " only CENTRAL, FORWARD, BACKWARD derivative are defined";
	}

	/*! \brief Calculate the position where the derivative is calculated
	 *
	 * In case of non staggered case this function just return a null grid_key, in case of staggered,
	 *  it calculate how the operator shift the calculation in the cell
	 *
	 * \param pos position where we are calculating the derivative
	 * \param gs Grid info
	 * \param s_pos staggered position of the properties
	 *
	 */
	inline static grid_key_dx<Sys_eqs::dims> position(grid_key_dx<Sys_eqs::dims> & pos, const grid_sm<Sys_eqs::dims,void> & gs, const openfpm::vector<comb<Sys_eqs::dims>> & s_pos = openfpm::vector<comb<Sys_eqs::dims>>())
	{
		std::cerr << "Error " << __FILE__ << ":" << __LINE__ << " only CENTRAL, FORWARD, BACKWARD derivative are defined";
	}
};

/*! \brief Central average scheme on direction i
 *
 * \verbatim
 *
 *  +0.5     +0.5
 *   *---+---*
 *
 * \endverbatim
 *
 */
template<unsigned int d, typename arg, typename Sys_eqs>
class Avg<d,arg,Sys_eqs,CENTRAL>
{
	public:

	/*! \brief Calculate which colums of the Matrix are non zero
	 *
	 * stub_or_real it is just for change the argument type when testing, in normal
	 * conditions it is a distributed map
	 *
	 * \param g_map It is the map explained in FDScheme
	 * \param kmap position where the average is calculated
	 * \param gs Grid info
	 * \param cols non-zero colums calculated by the function
	 * \param coeff coefficent (constant in front of the derivative)
	 *
	 * ### Example
	 *
	 * \snippet FDScheme_unit_tests.hpp Usage of stencil derivative
	 *
	 */
	inline static void value(const typename stub_or_real<Sys_eqs,Sys_eqs::dims,typename Sys_eqs::stype,typename Sys_eqs::b_grid::decomposition::extended_type>::type & g_map, grid_dist_key_dx<Sys_eqs::dims> & kmap , const grid_sm<Sys_eqs::dims,void> & gs, typename Sys_eqs::stype (& spacing )[Sys_eqs::dims] , std::unordered_map<long int,typename Sys_eqs::stype > & cols, typename Sys_eqs::stype coeff)
	{
		// if the system is staggered the CENTRAL derivative is equivalent to a forward derivative
		if (is_grid_staggered<Sys_eqs>::value())
		{
			Avg<d,arg,Sys_eqs,BACKWARD>::value(g_map,kmap,gs,spacing,cols,coeff);
			return;
		}

		long int old_val = kmap.getKeyRef().get(d);
		kmap.getKeyRef().set_d(d, kmap.getKeyRef().get(d) + 1);
		arg::value(g_map,kmap,gs,spacing,cols,coeff/2);
		kmap.getKeyRef().set_d(d,old_val);


		old_val = kmap.getKeyRef().get(d);
		kmap.getKeyRef().set_d(d, kmap.getKeyRef().get(d) - 1);
		arg::value(g_map,kmap,gs,spacing,cols,coeff/2);
		kmap.getKeyRef().set_d(d,old_val);
	}


	/*! \brief Calculate the position where the average is calculated
	 *
	 * It follow the same concept of central derivative
	 *
	 * \param pos position where we are calculating the derivative
	 * \param gs Grid info
	 * \param s_pos staggered position of the properties
	 *
	 */
	inline static grid_key_dx<Sys_eqs::dims> position(grid_key_dx<Sys_eqs::dims> & pos, const grid_sm<Sys_eqs::dims,void> & gs, const comb<Sys_eqs::dims> (& s_pos)[Sys_eqs::nvar])
	{
		auto arg_pos = arg::position(pos,gs,s_pos);
		if (is_grid_staggered<Sys_eqs>::value())
		{
			if (arg_pos.get(d) == -1)
			{
				arg_pos.set_d(d,0);
				return arg_pos;
			}
			else
			{
				arg_pos.set_d(d,-1);
				return arg_pos;
			}
		}

		return arg_pos;
	}
};

/*! \brief FORWARD average on direction i
 *
 * \verbatim
 *
 *  +0.5    0.5
 *    +------*
 *
 * \endverbatim
 *
 */
template<unsigned int d, typename arg, typename Sys_eqs>
class Avg<d,arg,Sys_eqs,FORWARD>
{
	public:

	/*! \brief Calculate which colums of the Matrix are non zero
	 *
	 * stub_or_real it is just for change the argument type when testing, in normal
	 * conditions it is a distributed map
	 *
	 * \param g_map It is the map explained in FDScheme
	 * \param kmap position where the average is calculated
	 * \param gs Grid info
	 * \param cols non-zero colums calculated by the function
	 * \param coeff coefficent (constant in front of the derivative)
	 *
	 * ### Example
	 *
	 * \snippet FDScheme_unit_tests.hpp Usage of stencil derivative
	 *
	 */
	inline static void value(const typename stub_or_real<Sys_eqs,Sys_eqs::dims,typename Sys_eqs::stype,typename Sys_eqs::b_grid::decomposition::extended_type>::type & g_map, grid_dist_key_dx<Sys_eqs::dims> & kmap , const grid_sm<Sys_eqs::dims,void> & gs, typename Sys_eqs::stype (& spacing )[Sys_eqs::dims] , std::unordered_map<long int,typename Sys_eqs::stype > & cols, typename Sys_eqs::stype coeff)
	{

		long int old_val = kmap.getKeyRef().get(d);
		kmap.getKeyRef().set_d(d, kmap.getKeyRef().get(d) + 1);
		arg::value(g_map,kmap,gs,spacing,cols,coeff/2);
		kmap.getKeyRef().set_d(d,old_val);

		// backward
		arg::value(g_map,kmap,gs,spacing,cols,coeff/2);
	}


	/*! \brief Calculate the position in the cell where the average is calculated
	 *
	 * In case of non staggered case this function just return a null grid_key, in case of staggered,
	 * the FORWARD scheme return the position of the staggered property
	 *
	 * \param pos position where we are calculating the derivative
	 * \param gs Grid info
	 * \param s_pos staggered position of the properties
	 *
	 */
	inline static grid_key_dx<Sys_eqs::dims> position(grid_key_dx<Sys_eqs::dims> & pos, const grid_sm<Sys_eqs::dims,void> & gs, const comb<Sys_eqs::dims> (& s_pos)[Sys_eqs::nvar])
	{
		return arg::position(pos,gs,s_pos);
	}
};

/*! \brief First order BACKWARD derivative on direction i
 *
 * \verbatim
 *
 *  +0.5    0.5
 *    *------+
 *
 * \endverbatim
 *
 */
template<unsigned int d, typename arg, typename Sys_eqs>
class Avg<d,arg,Sys_eqs,BACKWARD>
{
	public:

	/*! \brief Calculate which colums of the Matrix are non zero
	 *
	 * stub_or_real it is just for change the argument type when testing, in normal
	 * conditions it is a distributed map
	 *
	 * \param g_map It is the map explained in FDScheme
	 * \param kmap position where the average is calculated
	 * \param gs Grid info
	 * \param cols non-zero colums calculated by the function
	 * \param coeff coefficent (constant in front of the derivative)
	 *
	 * ### Example
	 *
	 * \snippet FDScheme_unit_tests.hpp Usage of stencil derivative
	 *
	 */
	inline static void value(const typename stub_or_real<Sys_eqs,Sys_eqs::dims,typename Sys_eqs::stype,typename Sys_eqs::b_grid::decomposition::extended_type>::type & g_map, grid_dist_key_dx<Sys_eqs::dims> & kmap , const grid_sm<Sys_eqs::dims,void> & gs, typename Sys_eqs::stype (& spacing )[Sys_eqs::dims], std::unordered_map<long int,typename Sys_eqs::stype > & cols, typename Sys_eqs::stype coeff)
	{
		long int old_val = kmap.getKeyRef().get(d);
		kmap.getKeyRef().set_d(d, kmap.getKeyRef().get(d) - 1);
		arg::value(g_map,kmap,gs,spacing,cols,coeff/2);
		kmap.getKeyRef().set_d(d,old_val);

		// forward
		arg::value(g_map,kmap,gs,spacing,cols,coeff/2);
	}


	/*! \brief Calculate the position in the cell where the average is calculated
	 *
	 * In case of non staggered case this function just return a null grid_key, in case of staggered,
	 * the BACKWARD scheme return the position of the staggered property
	 *
	 * \param pos position where we are calculating the derivative
	 * \param gs Grid info
	 * \param s_pos staggered position of the properties
	 *
	 */
	inline static grid_key_dx<Sys_eqs::dims> position(grid_key_dx<Sys_eqs::dims> & pos, const grid_sm<Sys_eqs::dims,void> & gs, const comb<Sys_eqs::dims> (& s_pos)[Sys_eqs::nvar])
	{
		return arg::position(pos,gs,s_pos);
	}
};

#endif /* OPENFPM_NUMERICS_SRC_FINITEDIFFERENCE_AVERAGE_HPP_ */
