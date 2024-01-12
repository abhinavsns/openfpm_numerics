/*
 * FDScheme_unit_tests.hpp
 *
 *  Created on: Oct 5, 2015
 *      Author: i-bird
 */

#ifndef OPENFPM_NUMERICS_SRC_FINITEDIFFERENCE_FDSCHEME_UNIT_TESTS_HPP_
#define OPENFPM_NUMERICS_SRC_FINITEDIFFERENCE_FDSCHEME_UNIT_TESTS_HPP_

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "FiniteDifference/Derivative.hpp"
#include "FiniteDifference/Laplacian.hpp"
#include "Decomposition/CartDecomposition.hpp"
#include "util/grid_dist_testing.hpp"
#include "FiniteDifference/Average.hpp"
#include "FiniteDifference/sum.hpp"
#include "eq.hpp"

constexpr unsigned int x = 0;
constexpr unsigned int y = 1;
constexpr unsigned int z = 2;
constexpr unsigned int V = 0;

struct sys_nn
{
	//! dimensionaly of the equation (2D problem 3D problem ...)
	static const unsigned int dims = 2;
	//! number of degree of freedoms
	static const unsigned int nvar = 1;

	static const unsigned int ord = EQS_FIELD;

	//! boundary at X and Y
	static const bool boundary[];

	//! type of space float, double, ...
	typedef float stype;

	//! Base grid
	typedef grid_dist_id<dims,stype,aggregate<float>,CartDecomposition<2,stype> > b_grid;

	//! specify that we are on testing
	typedef void testing;
};

const bool sys_nn::boundary[] = {NON_PERIODIC,NON_PERIODIC};

struct sys_pp
{
	//! dimensionaly of the equation (2D problem 3D problem ...)
	static const unsigned int dims = 2;
	//! number of degree of freedom in the system
	static const unsigned int nvar = 1;
	static const unsigned int ord = EQS_FIELD;

	//! boundary at X and Y
	static const bool boundary[];

	//! type of space float, double, ...
	typedef float stype;

	//! Base grid
	typedef grid_dist_id<dims,stype,aggregate<float>,CartDecomposition<2,stype> > b_grid;

	//! Indicate we are on testing
	typedef void testing;
};

const bool sys_pp::boundary[] = {PERIODIC,PERIODIC};

//////////////////////// STAGGERED CASE //////////////////////////////////

struct syss_nn
{
	//! dimensionaly of the equation (2D problem 3D problem ...)
	static const unsigned int dims = 2;
	//! Degree of freedom in the system
	static const unsigned int nvar = 1;

	static const unsigned int ord = EQS_FIELD;

	//! Indicate that the type of grid is staggered
	static const unsigned int grid_type = STAGGERED_GRID;

	//! boundary at X and Y
	static const bool boundary[];

	//! type of space float, double, ...
	typedef float stype;

	//! Base grid
	typedef grid_dist_id<dims,stype,aggregate<float>,CartDecomposition<2,stype> > b_grid;

	//! Indicate we are on testing
	typedef void testing;
};

const bool syss_nn::boundary[] = {NON_PERIODIC,NON_PERIODIC};

struct syss_pp
{
	//! dimensionaly of the equation (2D problem 3D problem ...)
	static const unsigned int dims = 2;
	//! number of fields in the system
	static const unsigned int nvar = 1;
	static const unsigned int ord = EQS_FIELD;

	//! Indicate the grid is staggered
	static const unsigned int grid_type = STAGGERED_GRID;

	//! boundary at X and Y
	static const bool boundary[];

	//! type of space float, double, ...
	typedef float stype;

	//! Base grid
	typedef grid_dist_id<dims,stype,aggregate<float>,CartDecomposition<2,stype> > b_grid;

	//! Indicate we are on testing
	typedef void testing;
};

const bool syss_pp::boundary[] = {PERIODIC,PERIODIC};

BOOST_AUTO_TEST_SUITE( fd_test )

// Derivative central, composed central derivative,
// and central bulk + one-sided non periodic

BOOST_AUTO_TEST_CASE( der_central_non_periodic)
{
	//! [Usage of stencil derivative]

	// grid size
	size_t sz[2]={16,16};

	// grid_dist_testing
	grid_dist_testing<2> g_map(sz);

	// grid_sm
	grid_sm<2,void> ginfo(sz);

	// spacing
	float spacing[2] = {0.5,0.3};

	// Create several keys
	grid_dist_key_dx<2> key11(0,grid_key_dx<2>(1,1));
	grid_dist_key_dx<2> key00(0,grid_key_dx<2>(0,0));
	grid_dist_key_dx<2> key22(0,grid_key_dx<2>(2,2));
	grid_dist_key_dx<2> key1515(0,grid_key_dx<2>(15,15));

	// filled colums
	std::unordered_map<long int,float> cols_x;
	std::unordered_map<long int,float> cols_y;

	D<x,Field<V,sys_nn>,sys_nn>::value(g_map,key11,ginfo,spacing,cols_x,1);
	D<y,Field<V,sys_nn>,sys_nn>::value(g_map,key11,ginfo,spacing,cols_y,1);

	BOOST_REQUIRE_EQUAL(cols_x.size(),2ul);
	BOOST_REQUIRE_EQUAL(cols_y.size(),2ul);

	BOOST_REQUIRE_EQUAL(cols_x[17+1],1/spacing[0]/2.0);
	BOOST_REQUIRE_EQUAL(cols_x[17-1],-1/spacing[0]/2.0);

	BOOST_REQUIRE_EQUAL(cols_y[17+16],1/spacing[1]/2.0);
	BOOST_REQUIRE_EQUAL(cols_y[17-16],-1/spacing[1]/2.0);

	//! [Usage of stencil derivative]

	// filled colums

	std::unordered_map<long int,float> cols_xx;
	std::unordered_map<long int,float> cols_xy;
	std::unordered_map<long int,float> cols_yx;
	std::unordered_map<long int,float> cols_yy;

	// Composed derivative

	D<x,D<x,Field<V,sys_nn>,sys_nn>,sys_nn>::value(g_map,key22,ginfo,spacing,cols_xx,1);
	D<x,D<y,Field<V,sys_nn>,sys_nn>,sys_nn>::value(g_map,key22,ginfo,spacing,cols_xy,1);
	D<y,D<x,Field<V,sys_nn>,sys_nn>,sys_nn>::value(g_map,key22,ginfo,spacing,cols_yx,1);
	D<y,D<y,Field<V,sys_nn>,sys_nn>,sys_nn>::value(g_map,key22,ginfo,spacing,cols_yy,1);

	BOOST_REQUIRE_EQUAL(cols_xx.size(),3ul);
	BOOST_REQUIRE_EQUAL(cols_xy.size(),4ul);
	BOOST_REQUIRE_EQUAL(cols_yx.size(),4ul);
	BOOST_REQUIRE_EQUAL(cols_yy.size(),3ul);

	BOOST_REQUIRE_EQUAL(cols_xx[32],1/spacing[0]/spacing[0]/2.0/2.0);
	BOOST_REQUIRE_EQUAL(cols_xx[34],-2/spacing[0]/spacing[0]/2.0/2.0);
	BOOST_REQUIRE_EQUAL(cols_xx[36],1/spacing[0]/spacing[0]/2.0/2.0);

	BOOST_REQUIRE_EQUAL(cols_xy[17],1/spacing[0]/spacing[1]/2.0/2.0);
	BOOST_REQUIRE_EQUAL(cols_xy[19],-1/spacing[0]/spacing[1]/2.0/2.0);
	BOOST_REQUIRE_EQUAL(cols_xy[49],-1/spacing[0]/spacing[1]/2.0/2.0);
	BOOST_REQUIRE_EQUAL(cols_xy[51],1/spacing[0]/spacing[1]/2.0/2.0);

	BOOST_REQUIRE_EQUAL(cols_yx[17],1/spacing[0]/spacing[1]/2.0/2.0);
	BOOST_REQUIRE_EQUAL(cols_yx[19],-1/spacing[0]/spacing[1]/2.0/2.0);
	BOOST_REQUIRE_EQUAL(cols_yx[49],-1/spacing[0]/spacing[1]/2.0/2.0);
	BOOST_REQUIRE_EQUAL(cols_xy[51],1/spacing[0]/spacing[1]/2.0/2.0);

	BOOST_REQUIRE_EQUAL(cols_yy[2],1/spacing[1]/spacing[1]/2.0/2.0);
	BOOST_REQUIRE_EQUAL(cols_yy[34],-2/spacing[1]/spacing[1]/2.0/2.0);
	BOOST_REQUIRE_EQUAL(cols_yy[66],1/spacing[1]/spacing[1]/2.0/2.0);

	// Non periodic with one sided

	cols_x.clear();
	cols_y.clear();

	D<x,Field<V,sys_nn>,sys_nn,CENTRAL_B_ONE_SIDE>::value(g_map,key11,ginfo,spacing,cols_x,1);
	D<y,Field<V,sys_nn>,sys_nn,CENTRAL_B_ONE_SIDE>::value(g_map,key11,ginfo,spacing,cols_y,1);

	BOOST_REQUIRE_EQUAL(cols_x.size(),2ul);
	BOOST_REQUIRE_EQUAL(cols_y.size(),2ul);

	BOOST_REQUIRE_EQUAL(cols_x[17+1],1/spacing[0]);
	BOOST_REQUIRE_EQUAL(cols_x[17-1],-1/spacing[0]);

	BOOST_REQUIRE_EQUAL(cols_y[17+16],1/spacing[1]);
	BOOST_REQUIRE_EQUAL(cols_y[17-16],-1/spacing[1]);

	// Border left

	cols_x.clear();
	cols_y.clear();

	D<x,Field<V,sys_nn>,sys_nn,CENTRAL_B_ONE_SIDE>::value(g_map,key00,ginfo,spacing,cols_x,1);
	D<y,Field<V,sys_nn>,sys_nn,CENTRAL_B_ONE_SIDE>::value(g_map,key00,ginfo,spacing,cols_y,1);

	BOOST_REQUIRE_EQUAL(cols_x.size(),3ul);
	BOOST_REQUIRE_EQUAL(cols_y.size(),3ul);

	BOOST_REQUIRE_EQUAL(cols_x[0],-1.5/spacing[0]);
	BOOST_REQUIRE_EQUAL(cols_x[1],2/spacing[0]);
	BOOST_REQUIRE_EQUAL(cols_x[2],-0.5/spacing[0]);

	BOOST_REQUIRE_CLOSE(cols_y[0],-1.5/spacing[1],0.001);
	BOOST_REQUIRE_CLOSE(cols_y[16],2/spacing[1],0.001);
	BOOST_REQUIRE_CLOSE(cols_y[32],-0.5/spacing[1],0.001);

	// Border Top right

	cols_x.clear();
	cols_y.clear();

	D<x,Field<V,sys_nn>,sys_nn,CENTRAL_B_ONE_SIDE>::value(g_map,key1515,ginfo,spacing,cols_x,1);
	D<y,Field<V,sys_nn>,sys_nn,CENTRAL_B_ONE_SIDE>::value(g_map,key1515,ginfo,spacing,cols_y,1);

	BOOST_REQUIRE_EQUAL(cols_x.size(),3ul);
	BOOST_REQUIRE_EQUAL(cols_y.size(),3ul);

	BOOST_REQUIRE_EQUAL(cols_x[15*16+15],1.5/spacing[0]);
	BOOST_REQUIRE_EQUAL(cols_x[15*16+14],-2/spacing[0]);
	BOOST_REQUIRE_EQUAL(cols_x[15*16+13],0.5/spacing[0]);

	BOOST_REQUIRE_CLOSE(cols_y[15*16+15],1.5/spacing[1],0.001);
	BOOST_REQUIRE_CLOSE(cols_y[14*16+15],-2/spacing[1],0.001);
	BOOST_REQUIRE_CLOSE(cols_y[13*16+15],0.5/spacing[1],0.001);
}

// Derivative forward and backward non periodic

BOOST_AUTO_TEST_CASE( der_forward_backward_non_periodic )
{
	// grid size
	size_t sz[2]={16,16};

	// spacing
	float spacing[2] = {0.5,0.3};

	// grid_dist_testing
	grid_dist_testing<2> g_map(sz);

	// grid_sm
	grid_sm<2,void> ginfo(sz);

	// Create several keys
	grid_dist_key_dx<2> key11(0,grid_key_dx<2>(1,1));
	grid_dist_key_dx<2> key00(0,grid_key_dx<2>(0,0));
	grid_dist_key_dx<2> key22(0,grid_key_dx<2>(2,2));
	grid_dist_key_dx<2> key1515(0,grid_key_dx<2>(15,15));

	// filled colums
	std::unordered_map<long int,float> cols_x;
	std::unordered_map<long int,float> cols_y;

	D<x,Field<V,sys_nn>,sys_nn,FORWARD>::value(g_map,key11,ginfo,spacing,cols_x,1);
	D<y,Field<V,sys_nn>,sys_nn,FORWARD>::value(g_map,key11,ginfo,spacing,cols_y,1);

	BOOST_REQUIRE_EQUAL(cols_x.size(),2ul);
	BOOST_REQUIRE_EQUAL(cols_y.size(),2ul);

	BOOST_REQUIRE_EQUAL(cols_x[17+1],1/spacing[0]);
	BOOST_REQUIRE_EQUAL(cols_x[17],-1/spacing[0]);

	BOOST_REQUIRE_EQUAL(cols_y[17+16],1/spacing[1]);
	BOOST_REQUIRE_EQUAL(cols_y[17],-1/spacing[1]);

	cols_x.clear();
	cols_y.clear();

	D<x,Field<V,sys_nn>,sys_nn,BACKWARD>::value(g_map,key11,ginfo,spacing,cols_x,1);
	D<y,Field<V,sys_nn>,sys_nn,BACKWARD>::value(g_map,key11,ginfo,spacing,cols_y,1);

	BOOST_REQUIRE_EQUAL(cols_x.size(),2ul);
	BOOST_REQUIRE_EQUAL(cols_y.size(),2ul);

	BOOST_REQUIRE_EQUAL(cols_x[17],1/spacing[0]);
	BOOST_REQUIRE_EQUAL(cols_x[17-1],-1/spacing[0]);

	BOOST_REQUIRE_EQUAL(cols_y[17],1/spacing[1]);
	BOOST_REQUIRE_EQUAL(cols_y[17-16],-1/spacing[1]);
}

// Average central non periodic

BOOST_AUTO_TEST_CASE( avg_central_non_periodic)
{
	// grid size
	size_t sz[2]={16,16};

	// spacing
	float spacing[2] = {0.5,0.3};

	// grid_dist_testing
	grid_dist_testing<2> g_map(sz);

	// grid_sm
	grid_sm<2,void> ginfo(sz);

	// Create several keys
	grid_dist_key_dx<2> key11(0,grid_key_dx<2>(1,1));
	grid_dist_key_dx<2> key00(0,grid_key_dx<2>(0,0));
	grid_dist_key_dx<2> key22(0,grid_key_dx<2>(2,2));
	grid_dist_key_dx<2> key1515(0,grid_key_dx<2>(15,15));

	// filled colums
	std::unordered_map<long int,float> cols_x;
	std::unordered_map<long int,float> cols_y;

	Avg<x,Field<V,sys_nn>,sys_nn>::value(g_map,key11,ginfo,spacing,cols_x,1);
	Avg<y,Field<V,sys_nn>,sys_nn>::value(g_map,key11,ginfo,spacing,cols_y,1);

	BOOST_REQUIRE_EQUAL(cols_x.size(),2ul);
	BOOST_REQUIRE_EQUAL(cols_y.size(),2ul);

	BOOST_REQUIRE_EQUAL(cols_x[17+1],0.5);
	BOOST_REQUIRE_EQUAL(cols_x[17-1],0.5);

	BOOST_REQUIRE_EQUAL(cols_y[17+16],0.5);
	BOOST_REQUIRE_EQUAL(cols_y[17-16],0.5);

	// filled colums

	std::unordered_map<long int,float> cols_xx;
	std::unordered_map<long int,float> cols_xy;
	std::unordered_map<long int,float> cols_yx;
	std::unordered_map<long int,float> cols_yy;

	// Composed derivative

	Avg<x,Avg<x,Field<V,sys_nn>,sys_nn>,sys_nn>::value(g_map,key22,ginfo,spacing,cols_xx,1);
	Avg<x,Avg<y,Field<V,sys_nn>,sys_nn>,sys_nn>::value(g_map,key22,ginfo,spacing,cols_xy,1);
	Avg<y,Avg<x,Field<V,sys_nn>,sys_nn>,sys_nn>::value(g_map,key22,ginfo,spacing,cols_yx,1);
	Avg<y,Avg<y,Field<V,sys_nn>,sys_nn>,sys_nn>::value(g_map,key22,ginfo,spacing,cols_yy,1);

	BOOST_REQUIRE_EQUAL(cols_xx.size(),3ul);
	BOOST_REQUIRE_EQUAL(cols_xy.size(),4ul);
	BOOST_REQUIRE_EQUAL(cols_yx.size(),4ul);
	BOOST_REQUIRE_EQUAL(cols_yy.size(),3ul);

	BOOST_REQUIRE_EQUAL(cols_xx[32],1/2.0/2.0);
	BOOST_REQUIRE_EQUAL(cols_xx[34],2/2.0/2.0);
	BOOST_REQUIRE_EQUAL(cols_xx[36],1/2.0/2.0);

	BOOST_REQUIRE_EQUAL(cols_xy[17],1/2.0/2.0);
	BOOST_REQUIRE_EQUAL(cols_xy[19],1/2.0/2.0);
	BOOST_REQUIRE_EQUAL(cols_xy[49],1/2.0/2.0);
	BOOST_REQUIRE_EQUAL(cols_xy[51],1/2.0/2.0);

	BOOST_REQUIRE_EQUAL(cols_yx[17],1/2.0/2.0);
	BOOST_REQUIRE_EQUAL(cols_yx[19],1/2.0/2.0);
	BOOST_REQUIRE_EQUAL(cols_yx[49],1/2.0/2.0);
	BOOST_REQUIRE_EQUAL(cols_xy[51],1/2.0/2.0);

	BOOST_REQUIRE_EQUAL(cols_yy[2],1/2.0/2.0);
	BOOST_REQUIRE_EQUAL(cols_yy[34],2/2.0/2.0);
	BOOST_REQUIRE_EQUAL(cols_yy[66],1/2.0/2.0);
}



BOOST_AUTO_TEST_CASE( der_central_staggered_non_periodic)
{
	// grid size
	size_t sz[2]={16,16};

	// spacing
	float spacing[2] = {0.5,0.3};

	// grid_sm
	grid_sm<2,void> ginfo(sz);

	// grid_dist_testing
	grid_dist_testing<2> g_map(sz);

	// Create several keys
	grid_dist_key_dx<2> key11(0,grid_key_dx<2>(1,1));
	grid_dist_key_dx<2> key22(0,grid_key_dx<2>(2,2));
	grid_dist_key_dx<2> key1515(0,grid_key_dx<2>(15,15));

	// filled colums
	std::unordered_map<long int,float> cols_x;
	std::unordered_map<long int,float> cols_y;

	D<x,Field<V,syss_nn>,syss_pp>::value(g_map,key11,ginfo,spacing,cols_x,1);
	D<y,Field<V,syss_nn>,syss_pp>::value(g_map,key11,ginfo,spacing,cols_y,1);

	BOOST_REQUIRE_EQUAL(cols_x.size(),2ul);
	BOOST_REQUIRE_EQUAL(cols_y.size(),2ul);

	BOOST_REQUIRE_EQUAL(cols_x[17],1/spacing[0]);
	BOOST_REQUIRE_EQUAL(cols_x[17-1],-1/spacing[0]);

	BOOST_REQUIRE_EQUAL(cols_y[17],1/spacing[1]);
	BOOST_REQUIRE_EQUAL(cols_y[17-16],-1/spacing[1]);

	cols_x.clear();
	cols_y.clear();
}

BOOST_AUTO_TEST_CASE( avg_central_staggered_non_periodic)
{
	// grid size
	size_t sz[2]={16,16};

	// spacing
	float spacing[2] = {0.5,0.3};

	// grid_sm
	grid_sm<2,void> ginfo(sz);

	// grid_dist_testing
	grid_dist_testing<2> g_map(sz);

	// Create several keys
	grid_dist_key_dx<2> key11(0,grid_key_dx<2>(1,1));
	grid_dist_key_dx<2> key22(0,grid_key_dx<2>(2,2));
	grid_dist_key_dx<2> key1515(0,grid_key_dx<2>(15,15));

	// filled colums
	std::unordered_map<long int,float> cols_x;
	std::unordered_map<long int,float> cols_y;

	Avg<x,Field<V,syss_pp>,syss_pp>::value(g_map,key11,ginfo,spacing,cols_x,1);
	Avg<y,Field<V,syss_pp>,syss_pp>::value(g_map,key11,ginfo,spacing,cols_y,1);

	BOOST_REQUIRE_EQUAL(cols_x.size(),2ul);
	BOOST_REQUIRE_EQUAL(cols_y.size(),2ul);

	BOOST_REQUIRE_EQUAL(cols_x[17],1/2.0);
	BOOST_REQUIRE_EQUAL(cols_x[17-1],1/2.0);

	BOOST_REQUIRE_EQUAL(cols_y[17],1/2.0);
	BOOST_REQUIRE_EQUAL(cols_y[17-16],1/2.0);

	cols_x.clear();
	cols_y.clear();
}

/////////////// Laplacian test

BOOST_AUTO_TEST_CASE( lap_periodic)
{
	//! [Laplacian usage]

	// grid size
	size_t sz[2]={16,16};

	// grid_sm
	grid_sm<2,void> ginfo(sz);

	// spacing
	float spacing[2] = {0.5,0.3};

	// grid_dist_testing
	grid_dist_testing<2> g_map(sz);

	// Create several keys
	grid_dist_key_dx<2> key11(0,grid_key_dx<2>(1,1));
	grid_dist_key_dx<2> key22(0,grid_key_dx<2>(2,2));
	grid_dist_key_dx<2> key1414(0,grid_key_dx<2>(14,14));

	// filled colums
	std::unordered_map<long int,float> cols;

	Lap<Field<V,sys_pp>,sys_pp>::value(g_map,key11,ginfo,spacing,cols,1);

	BOOST_REQUIRE_EQUAL(cols.size(),5ul);

	BOOST_REQUIRE_EQUAL(cols[1],1/spacing[1]/spacing[1]);
	BOOST_REQUIRE_EQUAL(cols[17-1],1/spacing[0]/spacing[0]);
	BOOST_REQUIRE_EQUAL(cols[17+1],1/spacing[0]/spacing[0]);
	BOOST_REQUIRE_EQUAL(cols[17+16],1/spacing[1]/spacing[1]);

	BOOST_REQUIRE_EQUAL(cols[17],-2/spacing[0]/spacing[0] - 2/spacing[1]/spacing[1]);

	//! [Laplacian usage]

	cols.clear();

	Lap<Field<V,sys_pp>,sys_pp>::value(g_map,key1414,ginfo,spacing,cols,1);

	BOOST_REQUIRE_EQUAL(cols.size(),5ul);

	BOOST_REQUIRE_EQUAL(cols[14*16+13],1.0/spacing[0]/spacing[0]);
	BOOST_REQUIRE_EQUAL(cols[14*16+15],1.0/spacing[0]/spacing[0]);
	BOOST_REQUIRE_CLOSE(cols[13*16+14],1.0/spacing[1]/spacing[1],0.001);
	BOOST_REQUIRE_CLOSE(cols[15*16+14],1.0/spacing[1]/spacing[1],0.001);

	BOOST_REQUIRE_EQUAL(cols[14*16+14],-2/spacing[0]/spacing[0] - 2/spacing[1]/spacing[1]);
}

// sum test


BOOST_AUTO_TEST_CASE( sum_periodic)
{
	//! [sum example]

	// grid size
	size_t sz[2]={16,16};

	// spacing
	float spacing[2] = {0.5,0.3};

	// grid_sm
	grid_sm<2,void> ginfo(sz);

	// grid_dist_testing
	grid_dist_testing<2> g_map(sz);

	// Create several keys
	grid_dist_key_dx<2> key11(0,grid_key_dx<2>(1,1));
	grid_dist_key_dx<2> key22(0,grid_key_dx<2>(2,2));
	grid_dist_key_dx<2> key1414(0,grid_key_dx<2>(14,14));

	// filled colums
	std::unordered_map<long int,float> cols;

	sum<Field<V,sys_pp>,Field<V,sys_pp>,sys_pp>::value(g_map,key11,ginfo,spacing,cols,1);

	BOOST_REQUIRE_EQUAL(cols.size(),1ul);

	BOOST_REQUIRE_EQUAL(cols[17],2);

	//! [sum example]

	cols.clear();

	sum<Field<V,sys_pp>, Field<V,sys_pp> , Field<V,sys_pp> ,sys_pp>::value(g_map,key11,ginfo,spacing,cols,1);

	BOOST_REQUIRE_EQUAL(cols.size(),1ul);

	BOOST_REQUIRE_EQUAL(cols[17],3);

	cols.clear();

	//! [minus example]

	sum<Field<V,sys_pp>, Field<V,sys_pp> , minus<Field<V,sys_pp>,sys_pp> ,sys_pp>::value(g_map,key11,ginfo,spacing,cols,1);

	BOOST_REQUIRE_EQUAL(cols.size(),1ul);

	BOOST_REQUIRE_EQUAL(cols[17],1ul);

	//! [minus example]
}

//////////////// Position ////////////////////

BOOST_AUTO_TEST_CASE( fd_test_use_staggered_position)
{
	// grid size
	size_t sz[2]={16,16};

	// grid_sm
	grid_sm<2,void> ginfo(sz);

	// Create a derivative row Matrix
	grid_key_dx<2> key00(0,0);
	grid_key_dx<2> key11(1,1);
	grid_key_dx<2> key22(2,2);
	grid_key_dx<2> key1515(15,15);

	comb<2> vx_c[] = {{(char)0,(char)-1}};
	comb<2> vy_c[] = {{(char)-1,(char)0}};

	grid_key_dx<2> key_ret_vx_x = D<x,Field<V,syss_nn>,syss_nn>::position(key00,ginfo,vx_c);
	grid_key_dx<2> key_ret_vx_y = D<y,Field<V,syss_nn>,syss_nn>::position(key00,ginfo,vx_c);
	grid_key_dx<2> key_ret_vy_x = D<x,Field<V,syss_nn>,syss_nn>::position(key00,ginfo,vy_c);
	grid_key_dx<2> key_ret_vy_y = D<y,Field<V,syss_nn>,syss_nn>::position(key00,ginfo,vy_c);

	BOOST_REQUIRE_EQUAL(key_ret_vx_x.get(0),0);
	BOOST_REQUIRE_EQUAL(key_ret_vx_x.get(1),0);
	BOOST_REQUIRE_EQUAL(key_ret_vx_y.get(0),-1);
	BOOST_REQUIRE_EQUAL(key_ret_vx_y.get(1),-1);
	BOOST_REQUIRE_EQUAL(key_ret_vy_y.get(0),0);
	BOOST_REQUIRE_EQUAL(key_ret_vy_y.get(1),0);
	BOOST_REQUIRE_EQUAL(key_ret_vy_x.get(0),-1);
	BOOST_REQUIRE_EQUAL(key_ret_vy_x.get(1),-1);

	// Composed derivative

	grid_key_dx<2> key_ret_xx = D<x,D<x,Field<V,syss_nn>,syss_nn>,syss_nn>::position(key00,ginfo,vx_c);
	grid_key_dx<2> key_ret_xy = D<x,D<y,Field<V,syss_nn>,syss_nn>,syss_nn>::position(key00,ginfo,vx_c);
	grid_key_dx<2> key_ret_yx = D<y,D<x,Field<V,syss_nn>,syss_nn>,syss_nn>::position(key00,ginfo,vx_c);
	grid_key_dx<2> key_ret_yy = D<y,D<y,Field<V,syss_nn>,syss_nn>,syss_nn>::position(key00,ginfo,vx_c);

	BOOST_REQUIRE_EQUAL(key_ret_xx.get(0),-1);
	BOOST_REQUIRE_EQUAL(key_ret_xx.get(1),0);

	BOOST_REQUIRE_EQUAL(key_ret_xy.get(0),0);
	BOOST_REQUIRE_EQUAL(key_ret_xy.get(1),-1);

	BOOST_REQUIRE_EQUAL(key_ret_yx.get(0),0);
	BOOST_REQUIRE_EQUAL(key_ret_yx.get(1),-1);

	BOOST_REQUIRE_EQUAL(key_ret_yy.get(0),-1);
	BOOST_REQUIRE_EQUAL(key_ret_yy.get(1),0);

	////////////////// Non periodic one sided

	key_ret_vx_x = D<x,Field<V,syss_nn>,syss_nn,CENTRAL_B_ONE_SIDE>::position(key00,ginfo,vx_c);
	key_ret_vx_y = D<y,Field<V,syss_nn>,syss_nn,CENTRAL_B_ONE_SIDE>::position(key00,ginfo,vx_c);
	key_ret_vy_x = D<x,Field<V,syss_nn>,syss_nn,CENTRAL_B_ONE_SIDE>::position(key00,ginfo,vy_c);
	key_ret_vy_y = D<y,Field<V,syss_nn>,syss_nn,CENTRAL_B_ONE_SIDE>::position(key00,ginfo,vy_c);

	BOOST_REQUIRE_EQUAL(key_ret_vx_x.get(0),-1);
	BOOST_REQUIRE_EQUAL(key_ret_vx_x.get(1),0);
	BOOST_REQUIRE_EQUAL(key_ret_vx_y.get(0),-1);
	BOOST_REQUIRE_EQUAL(key_ret_vx_y.get(1),0);
	BOOST_REQUIRE_EQUAL(key_ret_vy_y.get(0),0);
	BOOST_REQUIRE_EQUAL(key_ret_vy_y.get(1),-1);
	BOOST_REQUIRE_EQUAL(key_ret_vy_x.get(0),0);
	BOOST_REQUIRE_EQUAL(key_ret_vy_x.get(1),-1);

	// Border left*/
}

BOOST_AUTO_TEST_SUITE_END()


#endif /* OPENFPM_NUMERICS_SRC_FINITEDIFFERENCE_FDSCHEME_UNIT_TESTS_HPP_ */
