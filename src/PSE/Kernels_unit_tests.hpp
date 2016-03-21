/*
 * Kernels_unit_tests.hpp
 *
 *  Created on: Feb 17, 2016
 *      Author: i-bird
 */

#ifndef OPENFPM_NUMERICS_SRC_PSE_KERNELS_UNIT_TESTS_HPP_
#define OPENFPM_NUMERICS_SRC_PSE_KERNELS_UNIT_TESTS_HPP_

#include "PSE/Kernels_test_util.hpp"
#if defined(__GNUG__) && !defined(__clang__)
#include <boost/multiprecision/float128.hpp>
#endif

BOOST_AUTO_TEST_SUITE( pse_kernels_unit_tests )

BOOST_AUTO_TEST_CASE( pse_ker )
{
	Vcluster & v_cl = *global_v_cluster;

	// This test is not made to run in parallel
	if (v_cl.getProcessingUnits() > 1)
		return;

	openfpm::vector<openfpm::vector<double>> y;
	openfpm::vector<openfpm::vector<double>> y_res;

	// Load the result of the test
	y_res.load("test_data/PSE_convergence");

	// Every time increase the number of particles by 2
	for (size_t i = 250 ; i <= 2097152000 ; i*=2)
	{
		y.add();

		PSEError err;

		/////// Order 2 //////////////

#if defined(__GNUG__) && !defined(__clang__)

		PSE_test<boost::multiprecision::float128,Lap_PSE<1,boost::multiprecision::float128,2>>(i,2,err);
		y.last().add(err.linf_error);

		PSE_test<boost::multiprecision::float128,Lap_PSE<1,boost::multiprecision::float128,2>>(i,4,err);
		y.last().add(err.linf_error);

#endif

		PSE_test<double,Lap_PSE<1,double,2>>(i,2,err);
		y.last().add(err.linf_error);

		PSE_test<double,Lap_PSE<1,double,2>>(i,4,err);
		y.last().add(err.linf_error);

		PSE_test<float,Lap_PSE<1,float,2>>(i,2,err);
		y.last().add(err.linf_error);

		PSE_test<float,Lap_PSE<1,float,2>>(i,4,err);
		y.last().add(err.linf_error);

		//////// Order 4 /////////////

#if defined(__GNUG__) && !defined(__clang__)

		PSE_test<boost::multiprecision::float128,Lap_PSE<1,boost::multiprecision::float128,4>>(i,2,err);
		y.last().add(err.linf_error);

		PSE_test<boost::multiprecision::float128,Lap_PSE<1,boost::multiprecision::float128,4>>(i,4,err);
		y.last().add(err.linf_error);

		//////// Order 6 /////////////


		PSE_test<boost::multiprecision::float128,Lap_PSE<1,boost::multiprecision::float128,6>>(i,2,err);
		y.last().add(err.linf_error);

		PSE_test<boost::multiprecision::float128,Lap_PSE<1,boost::multiprecision::float128,6>>(i,4,err);
		y.last().add(err.linf_error);


		//////// Order 8 /////////////


		PSE_test<boost::multiprecision::float128,Lap_PSE<1,boost::multiprecision::float128,8>>(i,8,err);
		y.last().add(err.linf_error);

		PSE_test<boost::multiprecision::float128,Lap_PSE<1,boost::multiprecision::float128,8>>(i,16,err);
		y.last().add(err.linf_error);

#endif

	}

	// Check the result
	for (size_t i = 0 ; i < y.size(); i++)
	{
		for (size_t j = 0 ; j < y.get(i).size(); j++)
		{
			BOOST_REQUIRE_CLOSE(y.get(i).get(j),y_res.get(i).get(j),0.01);
		}
	}
}

BOOST_AUTO_TEST_SUITE_END()

#endif /* OPENFPM_NUMERICS_SRC_PSE_KERNELS_UNIT_TESTS_HPP_ */
