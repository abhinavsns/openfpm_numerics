//
// Created by Abhinav Singh on 17.05.20.
//

//
// Created by Abhinav Singh on 23.04.20.
//
#include "config.h"

#define BOOST_TEST_DYN_LINK

#include "util/util_debug.hpp"
#include <boost/test/unit_test.hpp>
#include <iostream>
#include "../DCPSE_op.hpp"
#include "../DCPSE_Solver.hpp"
#include "Operators/Vector/vector_dist_operators.hpp"
#include "Vector/vector_dist_subset.hpp"
#include "../EqnsStruct.hpp"

template<typename particle_type, typename particle_type2>
void indexUpdate(
        particle_type &Particles,
        particle_type2 &Part_subset,
        openfpm::vector<aggregate<int>> &up_p, openfpm::vector<aggregate<int>> &dw_p,
        openfpm::vector<aggregate<int>> &l_p, openfpm::vector<aggregate<int>> &r_p,
        openfpm::vector<aggregate<int>> &up_p1, openfpm::vector<aggregate<int>> &dw_p1,
        openfpm::vector<aggregate<int>> &l_p1, openfpm::vector<aggregate<int>> &r_p1,
        openfpm::vector<aggregate<int>> &corner_ul, openfpm::vector<aggregate<int>> &corner_ur,
        openfpm::vector<aggregate<int>> &corner_dl, openfpm::vector<aggregate<int>> &corner_dr,
        openfpm::vector<aggregate<int>> &bulk, Box<2, double> &up, Box<2, double> &down, Box<2, double> &left,
        Box<2, double> &right) {
    up_p.clear();
    dw_p.clear();
    l_p.clear();
    r_p.clear();
    up_p1.clear();
    dw_p1.clear();
    l_p1.clear();
    r_p1.clear();
    corner_ul.clear();
    corner_ur.clear();
    corner_dl.clear();
    corner_dr.clear();
    bulk.clear();
    Part_subset.clear();

    auto it2 = Particles.getDomainIterator();

    while (it2.isNext()) {
        auto p = it2.get();
        Point<2, double> xp = Particles.getPos(p);
        if (up.isInside(xp) == true) {
            if(left.isInside(xp) == true)
            {
                corner_ul.add();
                corner_ul.last().get<0>() = p.getKey();
            }
            else if (right.isInside(xp) == true)
            {
                corner_ur.add();
                corner_ur.last().get<0>() = p.getKey();
            }
            else
            {
                up_p1.add();
                up_p1.last().get<0>() = p.getKey();
            }
            up_p.add();
            up_p.last().get<0>() = p.getKey();

        } else if (down.isInside(xp) == true) {
            if(left.isInside(xp) == true)
            {
                corner_dl.add();
                corner_dl.last().get<0>() = p.getKey();
            }
            else if (right.isInside(xp) == true)
            {
                corner_dr.add();
                corner_dr.last().get<0>() = p.getKey();
            }
            else
            {
                dw_p1.add();
                dw_p1.last().get<0>() = p.getKey();
            }
            dw_p.add();
            dw_p.last().get<0>() = p.getKey();
        } else if (left.isInside(xp) == true) {
            if(up.isInside(xp) == false && down.isInside(xp) == false) {
                l_p1.add();
                l_p1.last().get<0>() = p.getKey();
            }
            l_p.add();
            l_p.last().get<0>() = p.getKey();
        } else if (right.isInside(xp) == true) {
            if(up.isInside(xp) == false && down.isInside(xp) == false) {
                r_p1.add();
                r_p1.last().get<0>() = p.getKey();
            }
            r_p.add();
            r_p.last().get<0>() = p.getKey();
        }
        else {
            bulk.add();
            bulk.last().get<0>() = p.getKey();
        }
        ++it2;
    }

    for (int i = 0; i < bulk.size(); i++) {
        Part_subset.add();
        Part_subset.getLastPos()[0] = Particles.getPos(bulk.template get<0>(i))[0];
        Part_subset.getLastPos()[1] = Particles.getPos(bulk.template get<0>(i))[1];
    }

}

BOOST_AUTO_TEST_SUITE(dcpse_op_suite_tests)

    BOOST_AUTO_TEST_CASE(Active2DPetsc) {
        timer tt2;
        tt2.start();
        const size_t sz[2] = {81, 81};
        Box<2, double> box({0, 0}, {10, 10});
        double Lx = box.getHigh(0);
        double Ly = box.getHigh(1);
        size_t bc[2] = {NON_PERIODIC, NON_PERIODIC};
        double spacing = box.getHigh(0) / (sz[0] - 1);
        double rCut = 3.1 * spacing;
        double rCut2 = 3.1 * spacing;
        int ord = 2;
        int ord2 = 2;
        double sampling_factor = 1.9;
        double sampling_factor2 = 1.9;
        double alpha_V = 1.0;
        double alpha_P = 1.0;
        Ghost<2, double> ghost(rCut);

        auto &v_cl = create_vcluster();

/*                                          pol                             V         vort                 Ext    Press     strain       stress                      Mfield,   dPol                      dV         RHS                  f1     f2     f3    f4     f5     f6       H               V_t      div    H_t                                                                                      delmu */
        vector_dist<2, double, aggregate<VectorS<2, double>, VectorS<2, double>, double[2][2], VectorS<2, double>, double, double[2][2], double[2][2], VectorS<2, double>, VectorS<2, double>, VectorS<2, double>, VectorS<2, double>, double, double, double, double, double, double, double, VectorS<2, double>, double, double, double[2], double[2], double[2], double[2], double[2], double[2], double, VectorS<2, double>, VectorS<2, double>, VectorS<2, double>, VectorS<2, double>, double,double,double>> Particles(
                0, box, bc, ghost);
        vector_dist<2, double, aggregate<double, double, VectorS<2, double>>> Particles_subset(
                Particles.getDecomposition(), 0);

        auto it = Particles.getGridIterator(sz);
        while (it.isNext()) {
            Particles.add();
            auto key = it.get();
            double x = key.get(0) * it.getSpacing(0);
            Particles.getLastPos()[0] = x;
            double y = key.get(1) * it.getSpacing(1);
            Particles.getLastPos()[1] = y;
            ++it;
        }

        Particles.map();
        Particles.ghost_get<0>();

        openfpm::vector<aggregate<int>> bulk;
        openfpm::vector<aggregate<int>> up_p;
        openfpm::vector<aggregate<int>> dw_p;
        openfpm::vector<aggregate<int>> l_p;
        openfpm::vector<aggregate<int>> r_p;

        constexpr int x = 0;
        constexpr int y = 1;

        constexpr int Polarization = 0;
        constexpr int Velocity = 1;
        constexpr int Vorticity = 2;
        constexpr int ExtForce = 3;
        constexpr int Pressure = 4;
        constexpr int Strain_rate = 5;
        constexpr int Stress = 6;
        constexpr int MolField = 7;
        auto Pos = getV<PROP_POS>(Particles);

        auto Pol = getV<Polarization>(Particles);
        auto V = getV<Velocity>(Particles);
        auto W = getV<Vorticity>(Particles);
        auto g = getV<ExtForce>(Particles);
        auto P = getV<Pressure>(Particles);
        auto P_bulk = getV<0>(Particles_subset); //Pressure only on inside
        auto u = getV<Strain_rate>(Particles);
        auto sigma = getV<Stress>(Particles);
        auto h = getV<MolField>(Particles);
        auto dPol = getV<8>(Particles);
        auto dV = getV<9>(Particles);
        auto RHS = getV<10>(Particles);
        auto f1 = getV<11>(Particles);
        auto f2 = getV<12>(Particles);
        auto f3 = getV<13>(Particles);
        auto f4 = getV<14>(Particles);
        auto f5 = getV<15>(Particles);
        auto f6 = getV<16>(Particles);
        auto H = getV<17>(Particles);
        auto H_bulk = getV<1>(Particles_subset); //Pressure only on inside
        auto Grad_bulk = getV<2>(Particles_subset);

        auto V_t = getV<18>(Particles);
        auto div = getV<19>(Particles);
        auto H_t = getV<20>(Particles);
        auto Df1 = getV<21>(Particles);
        auto Df2 = getV<22>(Particles);
        auto Df3 = getV<23>(Particles);
        auto Df4 = getV<24>(Particles);
        auto Df5 = getV<25>(Particles);
        auto Df6 = getV<26>(Particles);
        auto delmu = getV<27>(Particles);
        auto k1 = getV<28>(Particles);
        auto k2 = getV<29>(Particles);
        auto k3 = getV<30>(Particles);
        auto k4 = getV<31>(Particles);
        auto H_p_b = getV<32>(Particles);
        auto FranckEnergyDensity = getV<33>(Particles);
        auto r = getV<34>(Particles);


        double eta = 1.0;
        double nu = -0.5;
        double gama = 0.1;
        double zeta = 0.07;
        double Ks = 1.0;
        double Kb = 1.0;
        double lambda = 0.1;
        //double delmu = -1.0;
        g = 0;
        delmu = -1.0;
        P = 0;
        P_bulk = 0;
        V = 0;
        // Here fill up the boxes for particle boundary detection.
        Particles.ghost_get<ExtForce,28>();


        Box<2, double> up({box.getLow(0) - spacing / 2.0, box.getHigh(1) - spacing / 2.0},
                          {box.getHigh(0) + spacing / 2.0, box.getHigh(1) + spacing / 2.0});

        Box<2, double> down({box.getLow(0) - spacing / 2.0, box.getLow(1) - spacing / 2.0},
                            {box.getHigh(0) + spacing / 2.0, box.getLow(1) + spacing / 2.0});

        Box<2, double> left({box.getLow(0) - spacing / 2.0, box.getLow(1) + spacing / 2.0},
                            {box.getLow(0) + spacing / 2.0, box.getHigh(1) - spacing / 2.0});

        Box<2, double> right({box.getHigh(0) - spacing / 2.0, box.getLow(1) + spacing / 2.0},
                             {box.getHigh(0) + spacing / 2.0, box.getHigh(1) - spacing / 2.0});
        /*Box<2, double> mid({box.getHigh(0) / 2.0 - spacing, box.getHigh(1) / 2.0 - spacing / 2.0},
                           {box.getHigh(0) / 2.0, box.getHigh(1) / 2.0 + spacing / 2.0});
*/
        Box<2, double> mid({box.getLow(0) + 3.1 * spacing, box.getLow(1) + 3.1 * spacing},
                           {box.getHigh(0) - 3.1 * spacing, box.getHigh(1) - 3.1 * spacing});


        openfpm::vector<Box<2, double>> boxes;
        boxes.add(up);
        boxes.add(down);
        boxes.add(left);
        boxes.add(right);
        boxes.add(mid);

        VTKWriter<openfpm::vector<Box<2, double>>, VECTOR_BOX> vtk_box;
        vtk_box.add(boxes);
        vtk_box.write("vtk_box.vtk");

        auto it2 = Particles.getDomainIterator();

        while (it2.isNext()) {
            auto p = it2.get();
            Point<2, double> xp = Particles.getPos(p);
            Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * xp[x] - Lx) / Lx) - sin((2 * xp[y] - Ly) / Ly)));
            Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * xp[x] - Lx) / Lx) - sin((2 * xp[y] - Ly) / Ly)));
            if (up.isInside(xp) == true) {
                up_p.add();
                up_p.last().get<0>() = p.getKey();
            } else if (down.isInside(xp) == true) {
                dw_p.add();
                dw_p.last().get<0>() = p.getKey();
            } else if (left.isInside(xp) == true) {
                l_p.add();
                l_p.last().get<0>() = p.getKey();
            } else if (right.isInside(xp) == true) {
                r_p.add();
                r_p.last().get<0>() = p.getKey();
            } else {
                bulk.add();
                bulk.last().get<0>() = p.getKey();
            }
            ++it2;
        }


        for (int i = 0; i < bulk.size(); i++) {
            Particles_subset.add();
            Particles_subset.getLastPos()[0] = Particles.getPos(bulk.template get<0>(i))[0];
            Particles_subset.getLastPos()[1] = Particles.getPos(bulk.template get<0>(i))[1];
        }


        Particles_subset.map();
        Particles_subset.ghost_get<0>();

        Derivative_x Dx(Particles, ord, rCut, sampling_factor, support_options::RADIUS), Bulk_Dx(Particles_subset, ord,
                                                                                                 rCut2,
                                                                                                 sampling_factor2,
                                                                                                 support_options::RADIUS);
        Derivative_y Dy(Particles, ord, rCut, sampling_factor, support_options::RADIUS), Bulk_Dy(Particles_subset, ord2,
                                                                                                 rCut2,
                                                                                                 sampling_factor2,
                                                                                                 support_options::RADIUS);
        Derivative_xy Dxy(Particles, ord, rCut, sampling_factor, support_options::RADIUS);
        auto Dyx = Dxy;
        Derivative_xx Dxx(Particles, ord, rCut, sampling_factor,
                          support_options::RADIUS);//,Bulk_Dxx(Particles_subset, ord2, rCut2, sampling_factor2, support_options::RADIUS);
        Derivative_yy Dyy(Particles, ord, rCut, sampling_factor,
                          support_options::RADIUS);//,Bulk_Dyy(Particles_subset, ord2, rCut2, sampling_factor2, support_options::RADIUS);

        /*Derivative_x Dx(Particles, ord, rCut, sampling_factor), Bulk_Dx(Particles_subset, ord, rCut, sampling_factor2);
        Derivative_y Dy(Particles, ord, rCut, sampling_factor), Bulk_Dy(Particles_subset, ord, rCut, sampling_factor2);
        Derivative_xy Dxy(Particles, ord2, rCut2, sampling_factor2);
        auto Dyx = Dxy;
        Derivative_xx Dxx(Particles, ord2, rCut2, sampling_factor2),Bulk_Dxx(Particles_subset, ord2, rCut2, sampling_factor2);
        Derivative_yy Dyy(Particles, ord2, rCut2, sampling_factor2),Bulk_Dyy(Particles_subset, ord2, rCut2, sampling_factor2);*/

        petsc_solver<double> solverPetsc;
        solverPetsc.setSolver(KSPGMRES);
        //solverPetsc.setRestart(250);
        solverPetsc.setPreconditioner(PCJACOBI);
        petsc_solver<double> solverPetsc2;
        solverPetsc2.setSolver(KSPGMRES);
        solverPetsc2.setPreconditioner(PCJACOBI);


        eq_id vx, vy;
        timer tt;
        timer tt3;
        vx.setId(0);
        vy.setId(1);
        double V_err_eps = 5e-4;
        double V_err = 1, V_err_old;
        int n = 0;
        int nmax = 300;
        int ctr = 0, errctr;
        double dt = 3e-3;
        double tim = 0;
        double tf = 1e-1;
        div = 0;
        double sum, sum1, sum_k;
        while (tim <= tf) {
            tt.start();
            Particles.ghost_get<Polarization>();
            sigma[x][x] =
                    -Ks * Dx(Pol[x]) * Dx(Pol[x]) - Kb * Dx(Pol[y]) * Dx(Pol[y]) + (Kb - Ks) * Dy(Pol[x]) * Dx(Pol[y]);
            sigma[x][y] =
                    -Ks * Dy(Pol[y]) * Dx(Pol[y]) - Kb * Dy(Pol[x]) * Dx(Pol[x]) + (Kb - Ks) * Dx(Pol[y]) * Dx(Pol[x]);
            sigma[y][x] =
                    -Ks * Dx(Pol[x]) * Dy(Pol[x]) - Kb * Dx(Pol[y]) * Dy(Pol[y]) + (Kb - Ks) * Dy(Pol[x]) * Dy(Pol[y]);
            sigma[y][y] =
                    -Ks * Dy(Pol[y]) * Dy(Pol[y]) - Kb * Dy(Pol[x]) * Dy(Pol[x]) + (Kb - Ks) * Dx(Pol[y]) * Dy(Pol[x]);
            Particles.ghost_get<Stress>();


            h[y] = (Pol[x] * (Ks * Dyy(Pol[y]) + Kb * Dxx(Pol[y]) + (Ks - Kb) * Dxy(Pol[x])) -
                    Pol[y] * (Ks * Dxx(Pol[x]) + Kb * Dyy(Pol[x]) + (Ks - Kb) * Dxy(Pol[y])));

            Particles.ghost_get<MolField>();

            FranckEnergyDensity = (Ks / 2.0) * ((Dx(Pol[x]) * Dx(Pol[x])) + (Dy(Pol[x]) * Dy(Pol[x])) + (Dx(Pol[y]) * Dx(Pol[y])) + (Dy(Pol[y]) * Dy(Pol[y]))) + ((Kb - Ks) / 2.0) * ((Dx(Pol[y]) - Dy(Pol[x])) * (Dx(Pol[y]) - Dy(Pol[x])));
            Particles.ghost_get<33>();


            f1 = gama * nu * Pol[x] * Pol[x] * (Pol[x] * Pol[x] - Pol[y] * Pol[y]) /
                 (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            f2 = 2.0 * gama * nu * Pol[x] * Pol[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y]) /
                 (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            f3 = gama * nu * Pol[y] * Pol[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y]) /
                 (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            f4 = 2.0 * gama * nu * Pol[x] * Pol[x] * Pol[x] * Pol[y] / (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            f5 = 4.0 * gama * nu * Pol[x] * Pol[x] * Pol[y] * Pol[y] / (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            f6 = 2.0 * gama * nu * Pol[x] * Pol[y] * Pol[y] * Pol[y] / (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            Particles.ghost_get<11, 12, 13, 14, 15, 16>();
            Df1[x] = Dx(f1);
            Df2[x] = Dx(f2);
            Df3[x] = Dx(f3);
            Df4[x] = Dx(f4);
            Df5[x] = Dx(f5);
            Df6[x] = Dx(f6);

            Df1[y] = Dy(f1);
            Df2[y] = Dy(f2);
            Df3[y] = Dy(f3);
            Df4[y] = Dy(f4);
            Df5[y] = Dy(f5);
            Df6[y] = Dy(f6);
            Particles.ghost_get<21, 22, 23, 24, 25, 26>();


            dV[x] = -0.5 * Dy(h[y]) + zeta * Dx(delmu * Pol[x] * Pol[x]) + zeta * Dy(delmu * Pol[x] * Pol[y]) -
                    zeta * Dx(0.5 * delmu * (Pol[x] * Pol[x] + Pol[y] * Pol[y])) -
                    0.5 * nu * Dx(-2.0 * h[y] * Pol[x] * Pol[y])
                    - 0.5 * nu * Dy(h[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y])) - Dx(sigma[x][x]) - Dy(sigma[x][y]) -
                    g[x]
                    - 0.5 * nu * Dx(-gama * lambda * delmu * (Pol[x] * Pol[x] - Pol[y] * Pol[y]))
                    - 0.5 * Dy(-2.0 * gama * lambda * delmu * (Pol[x] * Pol[y]));


            dV[y] = -0.5 * Dx(-h[y]) + zeta * Dy(delmu * Pol[y] * Pol[y]) + zeta * Dx(delmu * Pol[x] * Pol[y]) -
                    zeta * Dy(0.5 * delmu * (Pol[x] * Pol[x] + Pol[y] * Pol[y])) -
                    0.5 * nu * Dy(2.0 * h[y] * Pol[x] * Pol[y])
                    - 0.5 * nu * Dx(h[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y])) - Dx(sigma[y][x]) - Dy(sigma[y][y]) -
                    g[y]
                    - 0.5 * nu * Dy(gama * lambda * delmu * (Pol[x] * Pol[x] - Pol[y] * Pol[y]))
                    - 0.5 * Dx(-2.0 * gama * lambda * delmu * (Pol[x] * Pol[y]));
            Particles.ghost_get<9>();


            //Particles.write("PolarI");
            //Velocity Solution n iterations


            auto Stokes1 = eta * (Dxx(V[x]) + Dyy(V[x]))
                           + 0.5 * nu * (Df1[x] * Dx(V[x]) + f1 * Dxx(V[x]))
                           + 0.5 * nu * (Df2[x] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f2 * 0.5 * (Dxx(V[y]) + Dyx(V[x])))
                           + 0.5 * nu * (Df3[x] * Dy(V[y]) + f3 * Dyx(V[y]))
                           + 0.5 * nu * (Df4[y] * Dx(V[x]) + f4 * Dxy(V[x]))
                           + 0.5 * nu * (Df5[y] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f5 * 0.5 * (Dxy(V[y]) + Dyy(V[x])))
                           + 0.5 * nu * (Df6[y] * Dy(V[y]) + f6 * Dyy(V[y]));
            auto Stokes2 = eta * (Dxx(V[y]) + Dyy(V[y]))
                           - 0.5 * nu * (Df1[y] * Dx(V[x]) + f1 * Dxy(V[x]))
                           - 0.5 * nu * (Df2[y] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f2 * 0.5 * (Dxy(V[y]) + Dyy(V[x])))
                           - 0.5 * nu * (Df3[y] * Dy(V[y]) + f3 * Dyy(V[y]))
                           + 0.5 * nu * (Df4[x] * Dx(V[x]) + f4 * Dxx(V[x]))
                           + 0.5 * nu * (Df5[x] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f5 * 0.5 * (Dxx(V[y]) + Dyx(V[x])))
                           + 0.5 * nu * (Df6[x] * Dy(V[y]) + f6 * Dyx(V[y]));
            tt.stop();
            std::cout << "Init of Velocity took " << tt.getwct() << " seconds." << std::endl;
            tt.start();
            V_err = 1;
            n = 0;
            errctr = 0;
            while (V_err >= V_err_eps && n <= nmax) {

                petsc_solver<double> solverPetsc;
                solverPetsc.setSolver(KSPGMRES);
                //solverPetsc.setRestart(250);
                solverPetsc.setPreconditioner(PCJACOBI);
                petsc_solver<double> solverPetsc2;
                solverPetsc2.setSolver(KSPGMRES);
                solverPetsc2.setPreconditioner(PCJACOBI);

                RHS[x] = dV[x];
                RHS[y] = dV[y];
                Particles_subset.ghost_get<0>();
                Grad_bulk[x] = Bulk_Dx(P_bulk);
                Grad_bulk[y] = Bulk_Dy(P_bulk);
                Particles_subset.ghost_get<2>();
                for (int i = 0; i < bulk.size(); i++) {
                    Particles.template getProp<10>(bulk.template get<0>(i))[x] += Particles_subset.getProp<2>(i)[x];
                    Particles.template getProp<10>(bulk.template get<0>(i))[y] += Particles_subset.getProp<2>(i)[y];
                }
                Particles.ghost_get<10>();
                DCPSE_scheme<equations2d2, decltype(Particles)> Solver(Particles);
                Solver.impose(Stokes1, bulk, RHS[0], vx);
                Solver.impose(Stokes2, bulk, RHS[1], vy);
                Solver.impose(V[x], up_p, 0, vx);
                Solver.impose(V[y], up_p, 0, vy);
                Solver.impose(V[x], dw_p, 0, vx);
                Solver.impose(V[y], dw_p, 0, vy);
                Solver.impose(V[x], l_p, 0, vx);
                Solver.impose(V[y], l_p, 0, vy);
                Solver.impose(V[x], r_p, 0, vx);
                Solver.impose(V[y], r_p, 0, vy);
                //Solver.solve(V[x], V[y]);
                Solver.solve_with_solver(solverPetsc, V[x], V[y]);
                Particles.ghost_get<Velocity>();
                div = -(Dx(V[x]) + Dy(V[y]));
                Particles.ghost_get<19>();
                auto Helmholtz = Dxx(H) + Dyy(H);
                DCPSE_scheme<equations2d1, decltype(Particles)> SolverH(Particles);
                SolverH.impose(Helmholtz, bulk, prop_id<19>());
                SolverH.impose(H, up_p, 0);
                SolverH.impose(H, dw_p, 0);
                SolverH.impose(H, l_p, 0);
                SolverH.impose(H, r_p, 0);
                //SolverH.solve(H);
                SolverH.solve_with_solver(solverPetsc2, H);
                //std::cout << "Helmholtz Solved in " << tt.getwct() << " seconds." << std::endl;
                Particles.ghost_get<17>();
                Particles.ghost_get<Velocity>();
                P = P + div;
                Particles.ghost_get<Pressure>();
                for (int i = 0; i < bulk.size(); i++) {
                    Particles_subset.getProp<0>(i) = Particles.template getProp<4>(bulk.template get<0>(i));
                }
                V[x] = V[x] + Dx(H);
                V[y] = V[y] + Dy(H);
                for (int j = 0; j < up_p.size(); j++) {
                    auto p = up_p.get<0>(j);
                    Particles.getProp<1>(p)[0] = 0;
                    Particles.getProp<1>(p)[1] = 0;
                    Particles.getProp<4>(p) = 0;

                }
                for (int j = 0; j < dw_p.size(); j++) {
                    auto p = dw_p.get<0>(j);
                    Particles.getProp<1>(p)[0] = 0;
                    Particles.getProp<1>(p)[1] = 0;
                    Particles.getProp<4>(p) = 0;
                }
                for (int j = 0; j < l_p.size(); j++) {
                    auto p = l_p.get<0>(j);
                    Particles.getProp<1>(p)[0] = 0;
                    Particles.getProp<1>(p)[1] = 0;
                    Particles.getProp<4>(p) = 0;
                }
                for (int j = 0; j < r_p.size(); j++) {
                    auto p = r_p.get<0>(j);
                    Particles.getProp<1>(p)[0] = 0;
                    Particles.getProp<1>(p)[1] = 0;
                    Particles.getProp<4>(p) = 0;
                }
                Particles.ghost_get<Velocity>();
                Particles.ghost_get<Pressure>();
                sum = 0;
                sum1 = 0;
                for (int j = 0; j < bulk.size(); j++) {
                    auto p = bulk.get<0>(j);
                    sum += (Particles.getProp<18>(p)[0] - Particles.getProp<1>(p)[0]) *
                           (Particles.getProp<18>(p)[0] - Particles.getProp<1>(p)[0]) +
                           (Particles.getProp<18>(p)[1] - Particles.getProp<1>(p)[1]) *
                           (Particles.getProp<18>(p)[1] - Particles.getProp<1>(p)[1]);
                    sum1 += Particles.getProp<1>(p)[0] * Particles.getProp<1>(p)[0] +
                            Particles.getProp<1>(p)[1] * Particles.getProp<1>(p)[1];
                }
                sum = sqrt(sum);
                sum1 = sqrt(sum1);

                v_cl.sum(sum);
                v_cl.sum(sum1);
                v_cl.execute();
                V_t = V;
                Particles.ghost_get<18>();
                V_err_old = V_err;
                V_err = sum / sum1;
                if (V_err > V_err_old) {
                    errctr++;
                    //alpha_P -= 0.1;
                } else {
                    errctr = 0;
                }
/*                if (errctr > 5)
                    break;*/
                n++;
                if (v_cl.rank() == 0) { std::cout << "Rel l2 cgs err in V = " << V_err << std::endl; }
            }
            tt.stop();
            std::cout << "Rel l2 cgs err in V = " << V_err << " and took " << tt.getwct() << " seconds with " << n
                      << " iterations."
                      << std::endl;

            u[x][x] = Dx(V[x]);
            u[x][y] = 0.5 * (Dx(V[y]) + Dy(V[x]));
            u[y][x] = 0.5 * (Dy(V[x]) + Dx(V[y]));
            u[y][y] = Dy(V[y]);

            W[x][x] = 0;
            W[x][y] = 0.5 * (Dy(V[x]) - Dx(V[y]));
            W[y][x] = 0.5 * (Dx(V[y]) - Dy(V[x]));
            W[y][y] = 0;


            h[x] = -gama * (lambda * delmu - nu * (u[x][x] * Pol[x] * Pol[x] +  u[y][y] * Pol[y] * Pol[y] + 2* u[x][y] * Pol[x] * Pol[y]) / (Pol[x] * Pol[x] + Pol[y] * Pol[y]));


            Particles.ghost_get<MolField, Strain_rate, Vorticity>();
            Particles.deleteGhost();
            Particles.write_frame("Polar_3e-3", ctr);
            Particles.ghost_get<MolField, Strain_rate, Vorticity>();
            ctr++;
            H_p_b = Pol[x] * Pol[x] + Pol[y] * Pol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<32>(p) =(Particles.getProp<32>(p)== 0) ? 1 : Particles.getProp<32>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<32>(p) =(Particles.getProp<32>(p)== 0) ? 1 : Particles.getProp<32>(p);
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<32>(p) =(Particles.getProp<32>(p)== 0) ? 1 : Particles.getProp<32>(p);
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<32>(p) =(Particles.getProp<32>(p)== 0) ? 1 : Particles.getProp<32>(p);
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<32>(p) =(Particles.getProp<32>(p)== 0) ? 1 : Particles.getProp<32>(p);
            }
            H_p_b=sqrt(H_p_b);

            k1[x] = ((h[x] * Pol[x] - h[y] * Pol[y]) / gama + lambda * delmu * Pol[x] -
                     nu * (u[x][x] * Pol[x] + u[x][y] * Pol[y]) + W[x][x] * Pol[x] +
                     W[x][y] * Pol[y]);// - V[x] * Dx(Pol[x]) - V[y] * Dy(Pol[x]));
            k1[y] = ((h[x] * Pol[y] + h[y] * Pol[x]) / gama + lambda * delmu * Pol[y] -
                     nu * (u[y][x] * Pol[x] + u[y][y] * Pol[y]) + W[y][x] * Pol[x] +
                     W[y][y] * Pol[y]);// - V[x] * Dx(Pol[y]) - V[y] * Dy(Pol[y]));
            Particles.ghost_get<28>();

            H_t = H_p_b;//+0.5*dt*(k1[x]*k1[x]+k1[y]*k1[y]);
            dPol = Pol + (0.5 * dt) * k1;
            dPol = dPol / H_t;
            Particles.ghost_get<8>();
            r=dPol[x]*dPol[x]+dPol[y]*dPol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }

            h[y] = (dPol[x] * (Ks * Dyy(dPol[y]) + Kb * Dxx(dPol[y]) + (Ks - Kb) * Dxy(dPol[x])) -
                    dPol[y] * (Ks * Dxx(dPol[x]) + Kb * Dyy(dPol[x]) + (Ks - Kb) * Dxy(dPol[y])));

            h[x] = -gama * (lambda * delmu - nu * ((u[x][x] * dPol[x] * dPol[x] +  u[y][y] * dPol[y] * dPol[y] + 2* u[x][y] * dPol[x] * dPol[y])/(r)));
            Particles.ghost_get<7>();


            k2[x] = ((h[x] * (dPol[x]) - h[y] * (dPol[y])) / gama +
                     lambda * delmu * (dPol[x]) -
                     nu * (u[x][x] * (dPol[x]) + u[x][y] * (dPol[y])) +
                     W[x][x] * (dPol[x]) + W[x][y] * (dPol[y])); //-V[x] * Dx((dPol[x])) - V[y] * Dy((dPol[x])));
            k2[y] = ((h[x] * (dPol[y]) + h[y] * (dPol[x])) / gama +
                     lambda * delmu * (dPol[y]) -
                     nu * (u[y][x] * (dPol[x]) + u[y][y] * (dPol[y])) +
                     W[y][x] * (dPol[x]) + W[y][y] * (dPol[y])); //-V[x] * Dx((dPol[y])) - V[y] * Dy((dPol[y])));
            Particles.ghost_get<29>();
            H_t = H_p_b;//+0.5*dt*(k2[x]*k2[x]+k2[y]*k2[y]);
            dPol = Pol + (0.5 * dt) * k2;
            dPol = dPol / H_t;
            Particles.ghost_get<8>();
            r=dPol[x]*dPol[x]+dPol[y]*dPol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }

            h[y] = (dPol[x] * (Ks * Dyy(dPol[y]) + Kb * Dxx(dPol[y]) + (Ks - Kb) * Dxy(dPol[x])) -
                    dPol[y] * (Ks * Dxx(dPol[x]) + Kb * Dyy(dPol[x]) + (Ks - Kb) * Dxy(dPol[y])));

            h[x] = -gama * (lambda * delmu - nu * ((u[x][x] * dPol[x] * dPol[x] +  u[y][y] * dPol[y] * dPol[y] + 2* u[x][y] * dPol[x] * dPol[y])/(r)));

            k3[x] = ((h[x] * (dPol[x]) - h[y] * (dPol[y])) / gama +
                     lambda * delmu * (dPol[x]) -
                     nu * (u[x][x] * (dPol[x]) + u[x][y] * (dPol[y])) +
                     W[x][x] * (dPol[x]) + W[x][y] * (dPol[y]));
            // -V[x] * Dx((dPol[x])) - V[y] * Dy((dPol[x])));
            k3[y] = ((h[x] * (dPol[y]) + h[y] * (dPol[x])) / gama +
                     lambda * delmu * (dPol[y]) -
                     nu * (u[y][x] * (dPol[x]) + u[y][y] * (dPol[y])) +
                     W[y][x] * (dPol[x]) + W[y][y] * (dPol[y]));
            // -V[x] * Dx((dPol[y])) - V[y] * Dy((dPol[y])));
            Particles.ghost_get<30>();
            H_t =  H_p_b;//+dt*(k3[x]*k3[x]+k3[y]*k3[y]);
            dPol = Pol + (dt * k3);
            dPol = dPol / H_t;
            Particles.ghost_get<8>();
            r=dPol[x]*dPol[x]+dPol[y]*dPol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }

            h[y] = (dPol[x] * (Ks * Dyy(dPol[y]) + Kb * Dxx(dPol[y]) + (Ks - Kb) * Dxy(dPol[x])) -
                    dPol[y] * (Ks * Dxx(dPol[x]) + Kb * Dyy(dPol[x]) + (Ks - Kb) * Dxy(dPol[y])));

            h[x] = -gama * (lambda * delmu - nu * ((u[x][x] * dPol[x] * dPol[x] +  u[y][y] * dPol[y] * dPol[y] + 2* u[x][y] * dPol[x] * dPol[y])/(r)));
            Particles.ghost_get<7>();


            k4[x] = ((h[x] * (dPol[x]) - h[y] * (dPol[y])) / gama +
                     lambda * delmu * (dPol[x]) -
                     nu * (u[x][x] * (dPol[x]) + u[x][y] * (dPol[y])) +
                     W[x][x] * (dPol[x]) +
                     W[x][y] * (dPol[y]));//   -V[x]*Dx( (dt * k3[x]+Pol[x])) -V[y]*Dy( (dt * k3[x]+Pol[x])));
            k4[y] = ((h[x] * (dPol[y]) + h[y] * (dPol[x])) / gama +
                     lambda * delmu * (dPol[y]) -
                     nu * (u[y][x] * (dPol[x]) + u[y][y] * (dPol[y])) +
                     W[y][x] * (dPol[x]) +
                     W[y][y] * (dPol[y]));//  -V[x]*Dx( (dt * k3[y]+Pol*[y])) -V[y]*Dy( (dt * k3[y]+Pol[y])));
            Particles.ghost_get<31>();

            Pol = Pol + (dt / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4);
            Pol = Pol / H_p_b;

            H_p_b = sqrt(Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            Pol = Pol / H_p_b;

            Pos = Pos + dt * V;
            Particles.map();
            Particles.ghost_get<0>();
            //indexUpdate(Particles, Particles_subset, up_p, dw_p, l_p, r_p,up_p1, dw_p1, l_p1, r_p1,corner_ul,corner_ur,corner_dl,corner_dr, bulk, up, down, left, right);
            Particles_subset.map();
            Particles_subset.ghost_get<0>();

            //Particles_subset.write("debug");
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }

            Particles.ghost_get<0, Vorticity, MolField>();
            Particles_subset.ghost_get<0,1,2>();

            tt.start();
            Dx.update(Particles);
            Dy.update(Particles);
            Dxy.update(Particles);
            auto Dyx = Dxy;
            Dxx.update(Particles);
            Dyy.update(Particles);

            Bulk_Dx.update(Particles_subset);
            Bulk_Dy.update(Particles_subset);

            tt.stop();
            std::cout << "Updation of operators took " << tt.getwct() << " seconds." << std::endl;

            std::cout << "Time step " << ctr << " : " << tim << " over." << std::endl;
            tim += dt;
            std::cout << "----------------------------------------------------------" << std::endl;
        }
        Particles.deleteGhost();
        tt2.stop();
        std::cout << "The simulation took " << tt2.getwct() << "Seconds.";
    }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




    BOOST_AUTO_TEST_CASE(Active2DEigen_multires) {
        timer tt2;
        tt2.start();
        const size_t sz[2] = {180,180};
        Box<2, double> box({0, 0}, {10, 10});
        double Lx = box.getHigh(0);
        double Ly = box.getHigh(1);
        size_t bc[2] = {NON_PERIODIC, NON_PERIODIC};
        double spacing = box.getHigh(0) / (sz[0] - 1);
        double rCut = 3.1 * spacing;
        double rCut2 = 3.1 * spacing;
        int ord = 2;
        int ord2 = 2;
        double sampling_factor = 3.4;
        double sampling_factor2 = 1.6;
        double alpha_V = 1.0;
        double alpha_P = 1.0;
        Ghost<2, double> ghost(rCut);

        auto &v_cl = create_vcluster();

/*                                          pol                             V         vort                 Ext    Press     strain       stress                      Mfield,   dPol                      dV         RHS                  f1     f2     f3    f4     f5     f6       H               V_t      div    H_t                                                                                      delmu */
        vector_dist<2, double, aggregate<VectorS<2, double>, VectorS<2, double>, double[2][2], VectorS<2, double>, double, double[2][2], double[2][2], VectorS<2, double>, VectorS<2, double>, VectorS<2, double>, VectorS<2, double>, double, double, double, double, double, double, double, VectorS<2, double>, double, double, double[2], double[2], double[2], double[2], double[2], double[2], double, VectorS<2, double>, VectorS<2, double>, VectorS<2, double>, VectorS<2, double>, double,double,double>> Particles(
                0, box, bc, ghost);
        vector_dist<2, double, aggregate<double, double, VectorS<2, double>>> Particles_subset(
                Particles.getDecomposition(), 0);
        double x0,y0,x1,y1;
        x0=10.0/(1+exp(0.8*(5)));
        y0=10.0/(1+exp(0.8*(5)));
        x1=10.0/(1+exp(0.8*(-5)));
        y1=10.0/(1+exp(0.8*(-5)));

        auto it = Particles.getGridIterator(sz);
        while (it.isNext()) {
            Particles.add();
            auto key = it.get();
            double x =key.get(0) * it.getSpacing(0);
            double y = key.get(1) * it.getSpacing(1);
            if(x==x0 || x==x1)
            {
                Particles.getLastPos()[0] = 10.0/(1+exp(0.8*sqrt((5-x)*(5-x)+(5-y)*(5-y))));
            }
            else
            {
                Particles.getLastPos()[0] = 10.0/(1+exp(0.8*sqrt((5-x)*(5-x)+(5-y)*(5-y))));
            }

            if(y==y0 || y==y1)
            {
                Particles.getLastPos()[1] = 10.0/(1+exp(0.8*sqrt((5-x)*(5-x)+(5-y)*(5-y))));
            }
            else {
                Particles.getLastPos()[1] = 10.0/(1+exp(0.8*sqrt((5-x)*(5-x)+(5-y)*(5-y))));
            }
            ++it;
        }
        spacing=10.0/(1+exp(5-2*it.getSpacing(0)));
        std::cout<<"Spacing"<<spacing<<std::endl;

        BOOST_TEST_MESSAGE("Sync domain across processors...");

        Particles.map();
        Particles.ghost_get<0>();

        Particles.write("Par");

        openfpm::vector<aggregate<int>> bulk;
        openfpm::vector<aggregate<int>> up_p;
        openfpm::vector<aggregate<int>> dw_p;
        openfpm::vector<aggregate<int>> l_p;
        openfpm::vector<aggregate<int>> r_p;
        openfpm::vector<aggregate<int>> up_p1;
        openfpm::vector<aggregate<int>> dw_p1;
        openfpm::vector<aggregate<int>> l_p1;
        openfpm::vector<aggregate<int>> r_p1;
        openfpm::vector<aggregate<int>> corner_ul;
        openfpm::vector<aggregate<int>> corner_ur;
        openfpm::vector<aggregate<int>> corner_dl;
        openfpm::vector<aggregate<int>> corner_dr;


        constexpr int x = 0;
        constexpr int y = 1;

        constexpr int Polarization = 0;
        constexpr int Velocity = 1;
        constexpr int Vorticity = 2;
        constexpr int ExtForce = 3;
        constexpr int Pressure = 4;
        constexpr int Strain_rate = 5;
        constexpr int Stress = 6;
        constexpr int MolField = 7;
        auto Pos = getV<PROP_POS>(Particles);

        auto Pol = getV<Polarization>(Particles);
        auto V = getV<Velocity>(Particles);
        auto W = getV<Vorticity>(Particles);
        auto g = getV<ExtForce>(Particles);
        auto P = getV<Pressure>(Particles);
        auto P_bulk = getV<0>(Particles_subset); //Pressure only on inside
        auto u = getV<Strain_rate>(Particles);
        auto sigma = getV<Stress>(Particles);
        auto h = getV<MolField>(Particles);
        auto dPol = getV<8>(Particles);
        auto dV = getV<9>(Particles);
        auto RHS = getV<10>(Particles);
        auto f1 = getV<11>(Particles);
        auto f2 = getV<12>(Particles);
        auto f3 = getV<13>(Particles);
        auto f4 = getV<14>(Particles);
        auto f5 = getV<15>(Particles);
        auto f6 = getV<16>(Particles);
        auto H = getV<17>(Particles);
        auto H_bulk = getV<1>(Particles_subset); //Pressure only on inside
        auto Grad_bulk = getV<2>(Particles_subset);

        auto V_t = getV<18>(Particles);
        auto div = getV<19>(Particles);
        auto H_t = getV<20>(Particles);
        auto Df1 = getV<21>(Particles);
        auto Df2 = getV<22>(Particles);
        auto Df3 = getV<23>(Particles);
        auto Df4 = getV<24>(Particles);
        auto Df5 = getV<25>(Particles);
        auto Df6 = getV<26>(Particles);
        auto delmu = getV<27>(Particles);
        auto k1 = getV<28>(Particles);
        auto k2 = getV<29>(Particles);
        auto k3 = getV<30>(Particles);
        auto k4 = getV<31>(Particles);
        auto H_p_b = getV<32>(Particles);
        auto FranckEnergyDensity = getV<33>(Particles);
        auto r = getV<34>(Particles);


        double eta = 1.0;
        double nu = -0.5;
        double gama = 0.1;
        double zeta = 0.07;
        double Ks = 1.0;
        double Kb = 1.0;
        double lambda = 0.1;
        //double delmu = -1.0;
        g = 0;
        delmu = -1.0;
        P = 0;
        P_bulk = 0;
        V = 0;
        // Here fill up the boxes for particle boundary detection.
        Particles.ghost_get<ExtForce,28>();


        Box<2, double> up({x0 - spacing / 2.0, y1 - spacing / 2.0},
                          {x1 + spacing / 2.0, y1 + spacing / 2.0});

        Box<2, double> down({x0 - spacing / 2.0, y0 - spacing / 2.0},
                            {x1 + spacing / 2.0,  y0 + spacing / 2.0});

        Box<2, double> left({x0 - spacing / 2.0, y0 - spacing / 2.0},
                            {x0 + spacing / 2.0,  y1 + spacing / 2.0});

        Box<2, double> right({x1 - spacing / 2.0, y0 - spacing / 2.0},
                             {x1 + spacing / 2.0,  y1 + spacing / 2.0});
        /*Box<2, double> mid({box.getHigh(0) / 2.0 - spacing, box.getHigh(1) / 2.0 - spacing / 2.0},
                           {box.getHigh(0) / 2.0, box.getHigh(1) / 2.0 + spacing / 2.0});
*/


        openfpm::vector<Box<2, double>> boxes;
        boxes.add(up);
        boxes.add(down);
        boxes.add(left);
        boxes.add(right);


        VTKWriter<openfpm::vector<Box<2, double>>, VECTOR_BOX> vtk_box;
        vtk_box.add(boxes);
        vtk_box.write("vtk_box.vtk");

        auto it2 = Particles.getDomainIterator();

        while (it2.isNext()) {
            auto p = it2.get();
            Point<2, double> xp = Particles.getPos(p);
            Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * xp[x] - Lx) / Lx) - sin((2 * xp[y] - Ly) / Ly)));
            Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * xp[x] - Lx) / Lx) - sin((2 * xp[y] - Ly) / Ly)));
            if (up.isInside(xp) == true) {
                if(left.isInside(xp) == true)
                {
                    corner_ul.add();
                    corner_ul.last().get<0>() = p.getKey();
                }
                else if (right.isInside(xp) == true)
                {
                    corner_ur.add();
                    corner_ur.last().get<0>() = p.getKey();
                }
                else
                {
                    up_p1.add();
                    up_p1.last().get<0>() = p.getKey();
                }
                up_p.add();
                up_p.last().get<0>() = p.getKey();

            } else if (down.isInside(xp) == true) {
                if(left.isInside(xp) == true)
                {
                    corner_dl.add();
                    corner_dl.last().get<0>() = p.getKey();
                }
                else if (right.isInside(xp) == true)
                {
                    corner_dr.add();
                    corner_dr.last().get<0>() = p.getKey();
                }
                else
                {
                    dw_p1.add();
                    dw_p1.last().get<0>() = p.getKey();
                }
                dw_p.add();
                dw_p.last().get<0>() = p.getKey();
            } else if (left.isInside(xp) == true) {
                if(up.isInside(xp) == false && down.isInside(xp) == false) {
                    l_p1.add();
                    l_p1.last().get<0>() = p.getKey();
                }
                l_p.add();
                l_p.last().get<0>() = p.getKey();
            } else if (right.isInside(xp) == true) {
                if(up.isInside(xp) == false && down.isInside(xp) == false) {
                    r_p1.add();
                    r_p1.last().get<0>() = p.getKey();
                }
                r_p.add();
                r_p.last().get<0>() = p.getKey();
            }
            else {
                bulk.add();
                bulk.last().get<0>() = p.getKey();
            }
            ++it2;
        }


        for (int i = 0; i < bulk.size(); i++) {
            Particles_subset.add();
            Particles_subset.getLastPos()[0] = Particles.getPos(bulk.template get<0>(i))[0];
            Particles_subset.getLastPos()[1] = Particles.getPos(bulk.template get<0>(i))[1];
        }


        Particles_subset.map();
        Particles_subset.ghost_get<0>();

        Particles_subset.write("Pars");
/*
        Derivative_x Dx(Particles, ord, rCut, sampling_factor, support_options::RADIUS), Bulk_Dx(Particles_subset, ord,
                                                                                                 rCut,
                                                                                                 sampling_factor,
                                                                                                 support_options::RADIUS);
        Derivative_y Dy(Particles, ord, rCut, sampling_factor, support_options::RADIUS), Bulk_Dy(Particles_subset, ord,
                                                                                                 rCut,
                                                                                                 sampling_factor,
                                                                                                 support_options::RADIUS);
        Derivative_xy Dxy(Particles, ord, rCut2, sampling_factor2, support_options::RADIUS);
        auto Dyx = Dxy;
        Derivative_xx Dxx(Particles, ord, rCut2, sampling_factor2,
                          support_options::RADIUS);//, Dxx2(Particles, ord2, rCut2, sampling_factor2, support_options::RADIUS);
        Derivative_yy Dyy(Particles, ord, rCut2, sampling_factor2,
                          support_options::RADIUS);//, Dyy2(Particles, ord2, rCut2, sampling_factor2, support_options::RADIUS);*/

        Derivative_x Dx(Particles, ord, rCut, sampling_factor), Bulk_Dx(Particles_subset, ord, rCut, sampling_factor);
        Derivative_y Dy(Particles, ord, rCut, sampling_factor), Bulk_Dy(Particles_subset, ord, rCut, sampling_factor);
        Derivative_xy Dxy(Particles, ord2, rCut, sampling_factor);
        auto Dyx = Dxy;
        Derivative_xx Dxx(Particles, ord2, rCut2, sampling_factor2),Bulk_Dxx(Particles_subset, ord2, rCut2, sampling_factor2);
        Derivative_yy Dyy(Particles, ord2, rCut2, sampling_factor2),Bulk_Dyy(Particles_subset, ord2, rCut2, sampling_factor2);

        petsc_solver<double> solverPetsc;
        solverPetsc.setSolver(KSPGMRES);
        //solverPetsc.setRestart(250);
        solverPetsc.setPreconditioner(PCJACOBI);
        petsc_solver<double> solverPetsc2;
        solverPetsc2.setSolver(KSPGMRES);
        solverPetsc2.setPreconditioner(PCJACOBI);


        eq_id vx, vy;
        timer tt;
        timer tt3;
        vx.setId(0);
        vy.setId(1);
        double V_err_eps = 1e-5 ;
        double V_err = 1, V_err_old;
        int n = 0;
        int nmax = 300;
        int ctr = 0, errctr;
        double dt = 2e-3    ;
        double tim = 0;
        double tf = 2e-1;
        div = 0;
        double sum, sum1, sum_k;
        while (tim <= tf) {
            tt.start();
            Particles.ghost_get<Polarization>();
            sigma[x][x] =
                    -Ks * Dx(Pol[x]) * Dx(Pol[x]) - Kb * Dx(Pol[y]) * Dx(Pol[y]) + (Kb - Ks) * Dy(Pol[x]) * Dx(Pol[y]);
            sigma[x][y] =
                    -Ks * Dy(Pol[y]) * Dx(Pol[y]) - Kb * Dy(Pol[x]) * Dx(Pol[x]) + (Kb - Ks) * Dx(Pol[y]) * Dx(Pol[x]);
            sigma[y][x] =
                    -Ks * Dx(Pol[x]) * Dy(Pol[x]) - Kb * Dx(Pol[y]) * Dy(Pol[y]) + (Kb - Ks) * Dy(Pol[x]) * Dy(Pol[y]);
            sigma[y][y] =
                    -Ks * Dy(Pol[y]) * Dy(Pol[y]) - Kb * Dy(Pol[x]) * Dy(Pol[x]) + (Kb - Ks) * Dx(Pol[y]) * Dy(Pol[x]);
            Particles.ghost_get<Stress>();


            h[y] = (Pol[x] * (Ks * Dyy(Pol[y]) + Kb * Dxx(Pol[y]) + (Ks - Kb) * Dxy(Pol[x])) -
                    Pol[y] * (Ks * Dxx(Pol[x]) + Kb * Dyy(Pol[x]) + (Ks - Kb) * Dxy(Pol[y])));

            Particles.ghost_get<MolField>();

            FranckEnergyDensity = (Ks / 2.0) * ((Dx(Pol[x]) * Dx(Pol[x])) + (Dy(Pol[x]) * Dy(Pol[x])) + (Dx(Pol[y]) * Dx(Pol[y])) + (Dy(Pol[y]) * Dy(Pol[y]))) + ((Kb - Ks) / 2.0) * ((Dx(Pol[y]) - Dy(Pol[x])) * (Dx(Pol[y]) - Dy(Pol[x])));
            Particles.ghost_get<33>();


            f1 = gama * nu * Pol[x] * Pol[x] * (Pol[x] * Pol[x] - Pol[y] * Pol[y]) /
                 (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            f2 = 2.0 * gama * nu * Pol[x] * Pol[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y]) /
                 (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            f3 = gama * nu * Pol[y] * Pol[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y]) /
                 (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            f4 = 2.0 * gama * nu * Pol[x] * Pol[x] * Pol[x] * Pol[y] / (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            f5 = 4.0 * gama * nu * Pol[x] * Pol[x] * Pol[y] * Pol[y] / (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            f6 = 2.0 * gama * nu * Pol[x] * Pol[y] * Pol[y] * Pol[y] / (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            Particles.ghost_get<11, 12, 13, 14, 15, 16>();
            Df1[x] = Dx(f1);
            Df2[x] = Dx(f2);
            Df3[x] = Dx(f3);
            Df4[x] = Dx(f4);
            Df5[x] = Dx(f5);
            Df6[x] = Dx(f6);

            Df1[y] = Dy(f1);
            Df2[y] = Dy(f2);
            Df3[y] = Dy(f3);
            Df4[y] = Dy(f4);
            Df5[y] = Dy(f5);
            Df6[y] = Dy(f6);
            Particles.ghost_get<21, 22, 23, 24, 25, 26>();


            dV[x] = -0.5 * Dy(h[y]) + zeta * Dx(delmu * Pol[x] * Pol[x]) + zeta * Dy(delmu * Pol[x] * Pol[y]) -
                    zeta * Dx(0.5 * delmu * (Pol[x] * Pol[x] + Pol[y] * Pol[y])) -
                    0.5 * nu * Dx(-2.0 * h[y] * Pol[x] * Pol[y])
                    - 0.5 * nu * Dy(h[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y])) - Dx(sigma[x][x]) - Dy(sigma[x][y]) -
                    g[x]
                    - 0.5 * nu * Dx(-gama * lambda * delmu * (Pol[x] * Pol[x] - Pol[y] * Pol[y]))
                    - 0.5 * Dy(-2.0 * gama * lambda * delmu * (Pol[x] * Pol[y]));


            dV[y] = -0.5 * Dx(-h[y]) + zeta * Dy(delmu * Pol[y] * Pol[y]) + zeta * Dx(delmu * Pol[x] * Pol[y]) -
                    zeta * Dy(0.5 * delmu * (Pol[x] * Pol[x] + Pol[y] * Pol[y])) -
                    0.5 * nu * Dy(2.0 * h[y] * Pol[x] * Pol[y])
                    - 0.5 * nu * Dx(h[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y])) - Dx(sigma[y][x]) - Dy(sigma[y][y]) -
                    g[y]
                    - 0.5 * nu * Dy(gama * lambda * delmu * (Pol[x] * Pol[x] - Pol[y] * Pol[y]))
                    - 0.5 * Dx(-2.0 * gama * lambda * delmu * (Pol[x] * Pol[y]));
            Particles.ghost_get<9>();


            //Particles.write("PolarI");
            //Velocity Solution n iterations


            auto Stokes1 = eta * (Dxx(V[x]) + Dyy(V[x]))
                           + 0.5 * nu * (Df1[x] * Dx(V[x]) + f1 * Dxx(V[x]))
                           + 0.5 * nu * (Df2[x] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f2 * 0.5 * (Dxx(V[y]) + Dyx(V[x])))
                           + 0.5 * nu * (Df3[x] * Dy(V[y]) + f3 * Dyx(V[y]))
                           + 0.5 * nu * (Df4[y] * Dx(V[x]) + f4 * Dxy(V[x]))
                           + 0.5 * nu * (Df5[y] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f5 * 0.5 * (Dxy(V[y]) + Dyy(V[x])))
                           + 0.5 * nu * (Df6[y] * Dy(V[y]) + f6 * Dyy(V[y]));
            auto Stokes2 = eta * (Dxx(V[y]) + Dyy(V[y]))
                           - 0.5 * nu * (Df1[y] * Dx(V[x]) + f1 * Dxy(V[x]))
                           - 0.5 * nu * (Df2[y] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f2 * 0.5 * (Dxy(V[y]) + Dyy(V[x])))
                           - 0.5 * nu * (Df3[y] * Dy(V[y]) + f3 * Dyy(V[y]))
                           + 0.5 * nu * (Df4[x] * Dx(V[x]) + f4 * Dxx(V[x]))
                           + 0.5 * nu * (Df5[x] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f5 * 0.5 * (Dxx(V[y]) + Dyx(V[x])))
                           + 0.5 * nu * (Df6[x] * Dy(V[y]) + f6 * Dyx(V[y]));
            tt.stop();
            std::cout << "Init of Velocity took " << tt.getwct() << " seconds." << std::endl;
            tt.start();
            V_err = 1;
            n = 0;
            errctr = 0;
            while (V_err >= V_err_eps && n <= nmax) {
                petsc_solver<double> solverPetsc;
                solverPetsc.setSolver(KSPGMRES);
                //solverPetsc.setRestart(250);
                solverPetsc.setPreconditioner(PCJACOBI);
                petsc_solver<double> solverPetsc2;
                solverPetsc2.setSolver(KSPGMRES);
                solverPetsc2.setPreconditioner(PCJACOBI);

                RHS[x] = dV[x];
                RHS[y] = dV[y];
                Particles_subset.ghost_get<0>();
                Grad_bulk[x] = Bulk_Dx(P_bulk);
                Grad_bulk[y] = Bulk_Dy(P_bulk);
                Particles_subset.ghost_get<2>();
                if (n%5==0)
                    Particles_subset.write_frame("Grad_debug",n );
                for (int i = 0; i < bulk.size(); i++) {
                    Particles.template getProp<10>(bulk.template get<0>(i))[x] += Particles_subset.getProp<2>(i)[x];
                    Particles.template getProp<10>(bulk.template get<0>(i))[y] += Particles_subset.getProp<2>(i)[y];
                }
                if (n%5==0)
                    Particles.write_frame("V_debug", n);
                Particles.ghost_get<10>();
                DCPSE_scheme<equations2d2, decltype(Particles)> Solver(Particles);
                Solver.impose(Stokes1, bulk, RHS[0], vx);
                Solver.impose(Stokes2, bulk, RHS[1], vy);
                Solver.impose(V[x], up_p, 0, vx);
                Solver.impose(V[y], up_p, 0, vy);
                Solver.impose(V[x], dw_p, 0, vx);
                Solver.impose(V[y], dw_p, 0, vy);
                Solver.impose(V[x], l_p, 0, vx);
                Solver.impose(V[y], l_p, 0, vy);
                Solver.impose(V[x], r_p, 0, vx);
                Solver.impose(V[y], r_p, 0, vy);
                //Solver.solve(V[x], V[y]);
                Solver.solve_with_solver(solverPetsc, V[x], V[y]);
                if (n%5==0)
                    Particles.write_frame("P_debug", n);
                Particles.ghost_get<Velocity>();
                /*for (int i = 0; i < bulk.size(); i++) {
                    Particles_subset.getProp<2>(i)[x] = Particles.template getProp<1>(bulk.template get<0>(i))[x];
                    Particles_subset.getProp<2>(i)[y] = Particles.template getProp<1>(bulk.template get<0>(i))[y];
                }
                P_bulk= (Bulk_Dx(Grad_bulk[x]) + Bulk_Dy(Grad_bulk[y]));
                for (int i = 0; i < bulk.size(); i++) {
                    Particles.template getProp<19>(bulk.template get<0>(i)) = Particles_subset.getProp<0>(i);
                }*/
                div = -(Dx(V[x]) + Dy(V[y]));
                Particles.ghost_get<19>();
                auto Helmholtz = Dxx(H) + Dyy(H);
                DCPSE_scheme<equations2d1, decltype(Particles)> SolverH(Particles);
                SolverH.impose(Helmholtz, bulk, prop_id<19>());
                SolverH.impose(H, up_p, 0);
                SolverH.impose(H, dw_p, 0);
                SolverH.impose(H, l_p, 0);
                SolverH.impose(H, r_p, 0);
/*                SolverH.impose(Dy(H), up_p1, 0);
                SolverH.impose(-Dy(H), dw_p1, 0);
                SolverH.impose(-Dx(H), l_p1, 0);
                SolverH.impose(Dx(H), r_p1, 0);*/
/*                SolverH.impose(Dy(H)-Dx(H), corner_ul, 0);
                SolverH.impose(Dx(H)+Dy(H), corner_ur, 0);

                SolverH.impose(-Dy(H)-Dx(H), corner_dl, 0);
                SolverH.impose(Dx(H)-Dy(H), corner_dr, 0);*/

                //SolverH.solve(H);
                SolverH.solve_with_solver(solverPetsc2, H);
                //V[x] = V[x] - Dx(H);
                //V[y] = V[y] - Dy(H);
                //std::cout << "Helmholtz Solved in " << tt.getwct() << " seconds." << std::endl;
                Particles.ghost_get<17>();
                Particles.ghost_get<Velocity>();
                P = P + div;
                Particles.ghost_get<Pressure>();
                for (int i = 0; i < bulk.size(); i++) {
                    Particles_subset.getProp<0>(i) = Particles.template getProp<4>(bulk.template get<0>(i));
                    //Particles_subset.getProp<1>(i) = Particles.template getProp<17>(bulk.template get<0>(i));
                }
                Particles_subset.ghost_get<1>();
                Grad_bulk[x] = Bulk_Dx(H_bulk);
                Grad_bulk[y] = Bulk_Dy(H_bulk);
                for (int i = 0; i < bulk.size(); i++) {
                    Particles.template getProp<1>(bulk.template get<0>(i))[x] += Particles_subset.getProp<2>(i)[x];
                    Particles.template getProp<1>(bulk.template get<0>(i))[y] += Particles_subset.getProp<2>(i)[y];
                }

                for (int j = 0; j < up_p.size(); j++) {
                    auto p = up_p.get<0>(j);
                    Particles.getProp<1>(p)[0] = 0;
                    Particles.getProp<1>(p)[1] = 0;
                    Particles.getProp<4>(p) = 0;

                }
                for (int j = 0; j < dw_p.size(); j++) {
                    auto p = dw_p.get<0>(j);
                    Particles.getProp<1>(p)[0] = 0;
                    Particles.getProp<1>(p)[1] = 0;
                    Particles.getProp<4>(p) = 0;
                }
                for (int j = 0; j < l_p.size(); j++) {
                    auto p = l_p.get<0>(j);
                    Particles.getProp<1>(p)[0] = 0;
                    Particles.getProp<1>(p)[1] = 0;
                    Particles.getProp<4>(p) = 0;
                }
                for (int j = 0; j < r_p.size(); j++) {
                    auto p = r_p.get<0>(j);
                    Particles.getProp<1>(p)[0] = 0;
                    Particles.getProp<1>(p)[1] = 0;
                    Particles.getProp<4>(p) = 0;
                }
                Particles.ghost_get<Velocity>();
                Particles.ghost_get<Pressure>();
                sum = 0;
                sum1 = 0;
                for (int j = 0; j < bulk.size(); j++) {
                    auto p = bulk.get<0>(j);
                    sum += (Particles.getProp<18>(p)[0] - Particles.getProp<1>(p)[0]) *
                           (Particles.getProp<18>(p)[0] - Particles.getProp<1>(p)[0]) +
                           (Particles.getProp<18>(p)[1] - Particles.getProp<1>(p)[1]) *
                           (Particles.getProp<18>(p)[1] - Particles.getProp<1>(p)[1]);
                    sum1 += Particles.getProp<1>(p)[0] * Particles.getProp<1>(p)[0] +
                            Particles.getProp<1>(p)[1] * Particles.getProp<1>(p)[1];
                }
                sum = sqrt(sum);
                sum1 = sqrt(sum1);

                v_cl.sum(sum);
                v_cl.sum(sum1);
                v_cl.execute();
                V_t = V;
                Particles.ghost_get<18>();
                V_err_old = V_err;
                V_err = sum / sum1;
                if (V_err > V_err_old) {
                    errctr++;
                    //alpha_P -= 0.1;
                } else {
                    errctr = 0;
                }
                if (errctr > 5)
                {
                    std::cout<<"CONVERGENCE LOOP BROKEN DUE TO INCREASE IN ERROR"<< std::endl;;
                    break;
                }
                n++;
                if (v_cl.rank() == 0)
                { std::cout << "Rel l2 cgs err in V = " << V_err <<" at "<<n<< std::endl; }
            }
            tt.stop();
            std::cout << "Rel l2 cgs err in V = " << V_err << " and took " << tt.getwct() << " seconds with " << n
                      << " iterations."
                      << std::endl;
            u[x][x] = Dx(V[x]);
            u[x][y] = 0.5 * (Dx(V[y]) + Dy(V[x]));
            u[y][x] = 0.5 * (Dy(V[x]) + Dx(V[y]));
            u[y][y] = Dy(V[y]);

            W[x][x] = 0;
            W[x][y] = 0.5 * (Dy(V[x]) - Dx(V[y]));
            W[y][x] = 0.5 * (Dx(V[y]) - Dy(V[x]));
            W[y][y] = 0;


            h[x] = -gama * (lambda * delmu - nu * (u[x][x] * Pol[x] * Pol[x] +  u[y][y] * Pol[y] * Pol[y] + 2* u[x][y] * Pol[x] * Pol[y]) / (Pol[x] * Pol[x] + Pol[y] * Pol[y]));


            Particles.ghost_get<MolField, Strain_rate, Vorticity>();
            Particles.deleteGhost();
            Particles.write_frame("Polar_3e-3", ctr);
            Particles.ghost_get<MolField, Strain_rate, Vorticity>();
            ctr++;
            H_p_b = Pol[x] * Pol[x] + Pol[y] * Pol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<32>(p) =(Particles.getProp<32>(p)== 0) ? 1 : Particles.getProp<32>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<32>(p) =(Particles.getProp<32>(p)== 0) ? 1 : Particles.getProp<32>(p);
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<32>(p) =(Particles.getProp<32>(p)== 0) ? 1 : Particles.getProp<32>(p);
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<32>(p) =(Particles.getProp<32>(p)== 0) ? 1 : Particles.getProp<32>(p);
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<32>(p) =(Particles.getProp<32>(p)== 0) ? 1 : Particles.getProp<32>(p);
            }
            H_p_b=sqrt(H_p_b);

            k1[x] = ((h[x] * Pol[x] - h[y] * Pol[y]) / gama + lambda * delmu * Pol[x] -
                     nu * (u[x][x] * Pol[x] + u[x][y] * Pol[y]) + W[x][x] * Pol[x] +
                     W[x][y] * Pol[y]);// - V[x] * Dx(Pol[x]) - V[y] * Dy(Pol[x]));
            k1[y] = ((h[x] * Pol[y] + h[y] * Pol[x]) / gama + lambda * delmu * Pol[y] -
                     nu * (u[y][x] * Pol[x] + u[y][y] * Pol[y]) + W[y][x] * Pol[x] +
                     W[y][y] * Pol[y]);// - V[x] * Dx(Pol[y]) - V[y] * Dy(Pol[y]));
            Particles.ghost_get<28>();

            H_t = H_p_b;//+0.5*dt*(k1[x]*k1[x]+k1[y]*k1[y]);
            dPol = Pol + (0.5 * dt) * k1;
            dPol = dPol / H_t;
            Particles.ghost_get<8>();
            r=dPol[x]*dPol[x]+dPol[y]*dPol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }

            h[y] = (dPol[x] * (Ks * Dyy(dPol[y]) + Kb * Dxx(dPol[y]) + (Ks - Kb) * Dxy(dPol[x])) -
                    dPol[y] * (Ks * Dxx(dPol[x]) + Kb * Dyy(dPol[x]) + (Ks - Kb) * Dxy(dPol[y])));

            h[x] = -gama * (lambda * delmu - nu * ((u[x][x] * dPol[x] * dPol[x] +  u[y][y] * dPol[y] * dPol[y] + 2* u[x][y] * dPol[x] * dPol[y])/(r)));
            Particles.ghost_get<7>();


            k2[x] = ((h[x] * (dPol[x]) - h[y] * (dPol[y])) / gama +
                     lambda * delmu * (dPol[x]) -
                     nu * (u[x][x] * (dPol[x]) + u[x][y] * (dPol[y])) +
                     W[x][x] * (dPol[x]) + W[x][y] * (dPol[y])); //-V[x] * Dx((dPol[x])) - V[y] * Dy((dPol[x])));
            k2[y] = ((h[x] * (dPol[y]) + h[y] * (dPol[x])) / gama +
                     lambda * delmu * (dPol[y]) -
                     nu * (u[y][x] * (dPol[x]) + u[y][y] * (dPol[y])) +
                     W[y][x] * (dPol[x]) + W[y][y] * (dPol[y])); //-V[x] * Dx((dPol[y])) - V[y] * Dy((dPol[y])));
            Particles.ghost_get<29>();
            H_t = H_p_b;//+0.5*dt*(k2[x]*k2[x]+k2[y]*k2[y]);
            dPol = Pol + (0.5 * dt) * k2;
            dPol = dPol / H_t;
            Particles.ghost_get<8>();
            r=dPol[x]*dPol[x]+dPol[y]*dPol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }

            h[y] = (dPol[x] * (Ks * Dyy(dPol[y]) + Kb * Dxx(dPol[y]) + (Ks - Kb) * Dxy(dPol[x])) -
                    dPol[y] * (Ks * Dxx(dPol[x]) + Kb * Dyy(dPol[x]) + (Ks - Kb) * Dxy(dPol[y])));

            h[x] = -gama * (lambda * delmu - nu * ((u[x][x] * dPol[x] * dPol[x] +  u[y][y] * dPol[y] * dPol[y] + 2* u[x][y] * dPol[x] * dPol[y])/(r)));

            k3[x] = ((h[x] * (dPol[x]) - h[y] * (dPol[y])) / gama +
                     lambda * delmu * (dPol[x]) -
                     nu * (u[x][x] * (dPol[x]) + u[x][y] * (dPol[y])) +
                     W[x][x] * (dPol[x]) + W[x][y] * (dPol[y]));
            // -V[x] * Dx((dPol[x])) - V[y] * Dy((dPol[x])));
            k3[y] = ((h[x] * (dPol[y]) + h[y] * (dPol[x])) / gama +
                     lambda * delmu * (dPol[y]) -
                     nu * (u[y][x] * (dPol[x]) + u[y][y] * (dPol[y])) +
                     W[y][x] * (dPol[x]) + W[y][y] * (dPol[y]));
            // -V[x] * Dx((dPol[y])) - V[y] * Dy((dPol[y])));
            Particles.ghost_get<30>();
            H_t =  H_p_b;//+dt*(k3[x]*k3[x]+k3[y]*k3[y]);
            dPol = Pol + (dt * k3);
            dPol = dPol / H_t;
            Particles.ghost_get<8>();
            r=dPol[x]*dPol[x]+dPol[y]*dPol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }

            h[y] = (dPol[x] * (Ks * Dyy(dPol[y]) + Kb * Dxx(dPol[y]) + (Ks - Kb) * Dxy(dPol[x])) -
                    dPol[y] * (Ks * Dxx(dPol[x]) + Kb * Dyy(dPol[x]) + (Ks - Kb) * Dxy(dPol[y])));

            h[x] = -gama * (lambda * delmu - nu * ((u[x][x] * dPol[x] * dPol[x] +  u[y][y] * dPol[y] * dPol[y] + 2* u[x][y] * dPol[x] * dPol[y])/(r)));
            Particles.ghost_get<7>();


            k4[x] = ((h[x] * (dPol[x]) - h[y] * (dPol[y])) / gama +
                     lambda * delmu * (dPol[x]) -
                     nu * (u[x][x] * (dPol[x]) + u[x][y] * (dPol[y])) +
                     W[x][x] * (dPol[x]) +
                     W[x][y] * (dPol[y]));//   -V[x]*Dx( (dt * k3[x]+Pol[x])) -V[y]*Dy( (dt * k3[x]+Pol[x])));
            k4[y] = ((h[x] * (dPol[y]) + h[y] * (dPol[x])) / gama +
                     lambda * delmu * (dPol[y]) -
                     nu * (u[y][x] * (dPol[x]) + u[y][y] * (dPol[y])) +
                     W[y][x] * (dPol[x]) +
                     W[y][y] * (dPol[y]));//  -V[x]*Dx( (dt * k3[y]+Pol*[y])) -V[y]*Dy( (dt * k3[y]+Pol[y])));
            Particles.ghost_get<31>();

            Pol = Pol + (dt / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4);
            Pol = Pol / H_p_b;

            H_p_b = sqrt(Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            Pol = Pol / H_p_b;

            Pos = Pos + dt * V;
            Particles.map();
            Particles.ghost_get<0>();
            indexUpdate(Particles, Particles_subset, up_p, dw_p, l_p, r_p,up_p1, dw_p1, l_p1, r_p1,corner_ul,corner_ur,corner_dl,corner_dr, bulk, up, down, left, right);
            Particles_subset.map();
            Particles_subset.ghost_get<0>();

            Particles_subset.write("debug");
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }

            Particles.ghost_get<0, Vorticity, MolField>();
            Particles_subset.ghost_get<0,1,2>();

            tt.start();
            Dx.update(Particles);
            Dy.update(Particles);
            Dxy.update(Particles);
            auto Dyx = Dxy;
            Dxx.update(Particles);
            Dyy.update(Particles);

            Bulk_Dx.update(Particles_subset);
            Bulk_Dy.update(Particles_subset);

            tt.stop();
            std::cout << "Updation of operators took " << tt.getwct() << " seconds." << std::endl;

            std::cout << "Time step " << ctr << " : " << tim << " over." << std::endl;
            tim += dt;
            std::cout << "----------------------------------------------------------" << std::endl;
        }
        Particles.deleteGhost();
        tt2.stop();
        std::cout << "The simulation took " << tt2.getwct() << "Seconds.";
    }


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    BOOST_AUTO_TEST_CASE(Active2DEigenP) {
        const size_t sz[2] = {21, 21};
        Box<2, double> box({0, 0}, {10, 10});
        double Lx = box.getHigh(0);
        double Ly = box.getHigh(1);
        size_t bc[2] = {PERIODIC, NON_PERIODIC};
        double spacing = box.getHigh(0) / (sz[0] - 1);
        int ord = 2;
        double rCut = 3.1 * spacing;
        double sampling_factor = 1.9;


        int ord2 = 2;
        double rCut2 = 3.1 * spacing;
        double sampling_factor2 = 1.9;

        Ghost<2, double> ghost(rCut);

/*                                          pol                             V         vort                 Ext    Press     strain       stress                      Mfield,   dPol                      dV         RHS                  f1     f2     f3    f4     f5     f6       H               V_t      div   H_t   */
        vector_dist<2, double, aggregate<VectorS<2, double>, VectorS<2, double>, double[2][2], VectorS<2, double>, double, double[2][2], double[2][2], VectorS<2, double>, VectorS<2, double>, VectorS<2, double>, VectorS<2, double>, double, double, double, double, double, double, double, VectorS<2, double>, double, double, double[2], double[2], double[2], double[2], double[2], double[2]>> Particles(
                0, box, bc, ghost);

        auto it = Particles.getGridIterator(sz);
        while (it.isNext()) {
            Particles.add();
            auto key = it.get();
            double x = key.get(0) * it.getSpacing(0);
            Particles.getLastPos()[0] = x;
            double y = key.get(1) * it.getSpacing(1);
            Particles.getLastPos()[1] = y * 0.99999;
            ++it;
        }

        Particles.map();
        Particles.ghost_get<0>();

        openfpm::vector<aggregate<int>> bulk;
        openfpm::vector<aggregate<int>> bulkP;
        openfpm::vector<aggregate<int>> up_p;
        openfpm::vector<aggregate<int>> dw_p;
        openfpm::vector<aggregate<int>> l_p;
        openfpm::vector<aggregate<int>> r_p;
        openfpm::vector<aggregate<int>> ref_p;

        constexpr int x = 0;
        constexpr int y = 1;

        constexpr int Polarization = 0;
        constexpr int Velocity = 1;
        constexpr int Vorticity = 2;
        constexpr int ExtForce = 3;
        constexpr int Pressure = 4;
        constexpr int Strain_rate = 5;
        constexpr int Stress = 6;
        constexpr int MolField = 7;

        auto Pol = getV<Polarization>(Particles);
        auto V = getV<Velocity>(Particles);
        V.setVarId(0);
        auto W = getV<Vorticity>(Particles);
        auto g = getV<ExtForce>(Particles);
        auto P = getV<Pressure>(Particles);
        P.setVarId(0);
        auto u = getV<Strain_rate>(Particles);
        auto sigma = getV<Stress>(Particles);
        auto h = getV<MolField>(Particles);
        auto dP = getV<8>(Particles);
        auto dV = getV<9>(Particles);
        auto RHS = getV<10>(Particles);
        auto f1 = getV<11>(Particles);
        auto f2 = getV<12>(Particles);
        auto f3 = getV<13>(Particles);
        auto f4 = getV<14>(Particles);
        auto f5 = getV<15>(Particles);
        auto f6 = getV<16>(Particles);
        auto H = getV<17>(Particles);
        auto V_t = getV<18>(Particles);
        auto div = getV<19>(Particles);
        auto H_t = getV<20>(Particles);
        auto Df1 = getV<21>(Particles);
        auto Df2 = getV<22>(Particles);
        auto Df3 = getV<23>(Particles);
        auto Df4 = getV<24>(Particles);
        auto Df5 = getV<25>(Particles);
        auto Df6 = getV<26>(Particles);


        double eta = 1.0;
        double nu = -0.5;
        double gama = 0.1;
        double zeta = 0.07;
        double Ks = 1.0;
        double Kb = 1.0;
        double lambda = 0.1;
        double delmu = -1.0;
        g = 0;
        V = 0;
        P = 0;

        // Here fill up the boxes for particle boundary detection.

        Box<2, double> up({box.getLow(0) - spacing / 2.0, box.getHigh(1) - spacing / 2.0},
                          {box.getHigh(0) + spacing / 2.0, box.getHigh(1) + spacing / 2.0});

        Box<2, double> down({box.getLow(0) - spacing / 2.0, box.getLow(1) - spacing / 2.0},
                            {box.getHigh(0) + spacing / 2.0, box.getLow(1) + spacing / 2.0});

        Box<2, double> left({box.getLow(0) - spacing / 2.0, box.getLow(1) + spacing / 2.0},
                            {box.getLow(0) + spacing / 2.0, box.getHigh(1) - spacing / 2.0});

        Box<2, double> right({box.getHigh(0) - spacing / 2.0, box.getLow(1) + spacing / 2.0},
                             {box.getHigh(0) + spacing / 2.0, box.getHigh(1) - spacing / 2.0});
        Box<2, double> mid({box.getHigh(0) / 2.0 - spacing, box.getHigh(1) / 2.0 - spacing / 2.0},
                           {box.getHigh(0) / 2.0, box.getHigh(1) / 2.0 + spacing / 2.0});

        openfpm::vector<Box<2, double>> boxes;
        boxes.add(up);
        boxes.add(down);
        boxes.add(left);
        boxes.add(right);
        boxes.add(mid);

        VTKWriter<openfpm::vector<Box<2, double>>, VECTOR_BOX> vtk_box;
        vtk_box.add(boxes);
        vtk_box.write("vtk_box.vtk");

        auto it2 = Particles.getDomainIterator();

        while (it2.isNext()) {
            auto p = it2.get();
            Point<2, double> xp = Particles.getPos(p);
            Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * xp[x] - Lx) / Lx) - sin((2 * xp[y] - Ly) / Ly)));
            Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * xp[x] - Lx) / Lx) - sin((2 * xp[y] - Ly) / Ly)));
            if (up.isInside(xp) == true) {
                up_p.add();
                up_p.last().get<0>() = p.getKey();
            } else if (down.isInside(xp) == true) {
                dw_p.add();
                dw_p.last().get<0>() = p.getKey();
            } else if (left.isInside(xp) == true) {
                l_p.add();
                l_p.last().get<0>() = p.getKey();
                bulk.add();
                bulk.last().get<0>() = p.getKey();
            } else if (right.isInside(xp) == true) {
                r_p.add();
                r_p.last().get<0>() = p.getKey();
                bulk.add();
                bulk.last().get<0>() = p.getKey();
            } else {
                if (mid.isInside(xp) == true) {
                    ref_p.add();
                    ref_p.last().get<0>() = p.getKey();
                    Particles.getProp<4>(p) = 0;
                } else {
                    bulkP.add();
                    bulkP.last().get<0>() = p.getKey();
                }
                bulk.add();
                bulk.last().get<0>() = p.getKey();
                // Particles.getProp<0>(p)[x]=  cos(2 * M_PI * (cos((2 * xp[x]- sz[x]) / sz[x]) - sin((2 * xp[y] - sz[y]) / sz[y])));
                // Particles.getProp<0>(p)[y] =  sin(2 * M_PI * (cos((2 * xp[x]- sz[x]) / sz[x]) - sin((2 * xp[y] - sz[y]) / sz[y])));
            }
            ++it2;
        }


        Derivative_x Dx(Particles, ord, rCut, sampling_factor, support_options::RADIUS), Dx2(Particles, ord2, rCut2,
                                                                                             sampling_factor2,
                                                                                             support_options::RADIUS);
        Derivative_y Dy(Particles, ord, rCut, sampling_factor, support_options::RADIUS), Dy2(Particles, ord2, rCut2,
                                                                                             sampling_factor2,
                                                                                             support_options::RADIUS);
        Derivative_xy Dxy(Particles, ord2, rCut2, sampling_factor2, support_options::RADIUS);
        auto Dyx = Dxy;
        Derivative_xx Dxx(Particles, ord2, rCut2, sampling_factor2, support_options::RADIUS);
        Derivative_yy Dyy(Particles, ord2, rCut2, sampling_factor2, support_options::RADIUS);
        Gradient Grad(Particles, ord, rCut, sampling_factor, support_options::RADIUS);
        Laplacian Lap(Particles, ord2, rCut2, sampling_factor2, support_options::RADIUS);
        Advection Adv(Particles, ord, rCut, sampling_factor, support_options::RADIUS);
        Divergence Div(Particles, ord, rCut, sampling_factor, support_options::RADIUS);

        Particles.ghost_get<Polarization>();
        sigma[x][x] =
                -Ks * Dx(Pol[x]) * Dx(Pol[x]) - Kb * Dx(Pol[y]) * Dx(Pol[y]) + (Kb - Ks) * Dy(Pol[x]) * Dx(Pol[y]);
        sigma[x][y] =
                -Ks * Dy(Pol[y]) * Dx(Pol[y]) - Kb * Dy(Pol[x]) * Dx(Pol[x]) + (Kb - Ks) * Dx(Pol[y]) * Dx(Pol[x]);
        sigma[y][x] =
                -Ks * Dx(Pol[x]) * Dy(Pol[x]) - Kb * Dx(Pol[y]) * Dy(Pol[y]) + (Kb - Ks) * Dy(Pol[x]) * Dy(Pol[y]);
        sigma[y][y] =
                -Ks * Dy(Pol[y]) * Dy(Pol[y]) - Kb * Dy(Pol[x]) * Dy(Pol[x]) + (Kb - Ks) * Dy(Pol[x]) * Dx(Pol[y]);
        Particles.ghost_get<Stress>();


        h[y] = Pol[x] * (Ks * Dyy(Pol[y]) + Kb * Dxx(Pol[y]) + (Ks - Kb) * Dxy(Pol[x])) -
               Pol[y] * (Ks * Dxx(Pol[x]) + Kb * Dyy(Pol[x]) + (Ks - Kb) * Dxy(Pol[y]));
        Particles.ghost_get<MolField>();


        f1 = gama * nu * Pol[x] * Pol[x] * (Pol[x] * Pol[x] - Pol[y] * Pol[y]) / (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
        f2 = 2.0 * gama * nu * Pol[x] * Pol[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y]) /
             (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
        f3 = gama * nu * Pol[y] * Pol[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y]) / (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
        f4 = 2.0 * gama * nu * Pol[x] * Pol[x] * Pol[x] * Pol[y] / (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
        f5 = 4.0 * gama * nu * Pol[x] * Pol[x] * Pol[y] * Pol[y] / (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
        f6 = 2.0 * gama * nu * Pol[x] * Pol[x] * Pol[x] * Pol[y] / (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
        Particles.ghost_get<11, 12, 13, 14, 15, 16>();
        Df1[x] = Dx(f1);
        Df2[x] = Dx(f2);
        Df3[x] = Dx(f3);
        Df4[x] = Dx(f4);
        Df5[x] = Dx(f5);
        Df6[x] = Dx(f6);
        Df1[y] = Dy(f1);
        Df2[y] = Dy(f2);
        Df3[y] = Dy(f3);
        Df4[y] = Dy(f4);
        Df5[y] = Dy(f5);
        Df6[y] = Dy(f6);
        Particles.ghost_get<21, 22, 23, 24, 25, 26>();


        dV[x] = -0.5 * Dy(h[y])
                + zeta * Dx(delmu * Pol[x] * Pol[x])
                + zeta * Dy(delmu * Pol[x] * Pol[y])
                - zeta * Dx(0.5 * delmu * (Pol[x] * Pol[x] + Pol[y] * Pol[y]))
                - 0.5 * nu * Dx(-2 * h[y] * Pol[x] * Pol[y])
                - 0.5 * nu * Dy(h[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y])) - Dx(sigma[x][x]) - Dy(sigma[x][y]) - g[x]
                - 0.5 * nu * Dx(-gama * lambda * delmu * (Pol[x] * Pol[x] - Pol[y] * Pol[y])) -
                0.5 * Dy(-2 * gama * lambda * delmu * (Pol[x] * Pol[y]));


        dV[y] = -0.5 * Dx(-h[y]) + zeta * Dy(delmu * Pol[y] * Pol[y]) + zeta * Dx(delmu * Pol[x] * Pol[y]) -
                zeta * Dy(0.5 * delmu * (Pol[x] * Pol[x] + Pol[y] * Pol[y])) -
                0.5 * nu * Dy(-2 * h[y] * Pol[x] * Pol[y])
                - 0.5 * nu * Dx(h[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y])) - Dx(sigma[y][x]) - Dy(sigma[y][y]) - g[y]
                - 0.5 * nu * Dy(gama * lambda * delmu * (Pol[x] * Pol[x] - Pol[y] * Pol[y])) -
                0.5 * Dx(-2 * gama * lambda * delmu * (Pol[x] * Pol[y]));
        Particles.ghost_get<9>();


        Particles.write_frame("Polar", 0);
        //Velocity Solution n iterations
        eq_id vx, vy;
        timer tt;
        vx.setId(0);
        vy.setId(1);
        double sum = 0, sum1 = 0;
        int n = 100;
        /* auto Stokes1 = nu * Lap(V[x]) + 0.5 * nu * (f1 * Dxx(V[x]) + Dx(f1) * Dx(V[x])) +
                        (Dx(f2) * 0.5 * (Dx(V[y]) + Dy(V[x])) + f2 * 0.5 * (Dxx(V[y]) + Dyx(V[x]))) +
                        (Dx(f3) * Dy(V[y]) + f3 * Dyx(V[y])) + (Dy(f4) * Dx(V[x]) + f4 * Dxy(V[x])) +
                        (Dy(f5) * 0.5 * (Dx(V[y]) + Dy(V[x])) + f5 * 0.5 * (Dxy(V[y]) + Dyy(V[x]))) +
                        (Dy(f6) * Dy(V[y]) + f6 * Dyy(V[y]));
         auto Stokes2 = nu * Lap(V[y]) + 0.5 * nu * (f1 * Dxy(V[x]) + Dy(f1) * Dx(V[x])) +
                        (Dy(f2) * 0.5 * (Dx(V[y]) + Dy(V[x])) + f2 * 0.5 * (Dxy(V[y]) + Dyy(V[x]))) +
                        (Dy(f3) * Dy(V[y]) + f3 * Dyy(V[y])) + (Dx(f4) * Dx(V[x]) + f4 * Dxx(V[x])) +
                        (Dx(f5) * 0.5 * (Dx(V[y]) + Dy(V[x])) + f5 * 0.5 * (Dxx(V[y]) + Dyx(V[x]))) +
                        (Dx(f6) * Dy(V[y]) + f6 * Dyx(V[y]));
 */
        auto Stokes1 = eta * Lap(V[x]) + 0.5 * nu * (f1 * Dxx(V[x]) + Df1[x] * Dx(V[x])) +
                       (Df2[x] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f2 * 0.5 * (Dxx(V[y]) + Dyx(V[x]))) +
                       (Df3[x] * Dy(V[y]) + f3 * Dyx(V[y])) + (Df4[y] * Dx(V[x]) + f4 * Dxy(V[x])) +
                       (Df5[y] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f5 * 0.5 * (Dxy(V[y]) + Dyy(V[x]))) +
                       (Df6[y] * Dy(V[y]) + f6 * Dyy(V[y]));
        auto Stokes2 = eta * Lap(V[y]) + 0.5 * nu * (f1 * Dxy(V[x]) + Df1[y] * Dx(V[x])) +
                       (Df2[y] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f2 * 0.5 * (Dxy(V[y]) + Dyy(V[x]))) +
                       (Df3[y] * Dy(V[y]) + f3 * Dyy(V[y])) + (Df4[x] * Dx(V[x]) + f4 * Dxx(V[x])) +
                       (Df5[x] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f5 * 0.5 * (Dxx(V[y]) + Dyx(V[x]))) +
                       (Df6[x] * Dy(V[y]) + f6 * Dyx(V[y]));

        auto Helmholtz = Lap(H);
        auto D_y = Dy2(H);
        auto D_x = Dx2(H);

        for (int i = 1; i <= n; i++) {
            RHS[x] = Dx(P) + dV[x];
            RHS[y] = Dy(P) + dV[y];
            Particles.ghost_get<10>();
            DCPSE_scheme<equations2d2pE, decltype(Particles)> Solver(Particles);
            Solver.impose(Stokes1, bulk, RHS[0], vx);
            Solver.impose(Stokes2, bulk, RHS[1], vy);
            Solver.impose(V[x], up_p, 0, vx);
            Solver.impose(V[y], up_p, 0, vy);
            Solver.impose(V[x], dw_p, 0, vx);
            Solver.impose(V[y], dw_p, 0, vy);
            tt.start();
            Solver.solve(V[x], V[y]);
            tt.stop();
            std::cout << "Stokes Solved in " << tt.getwct() << " seconds." << std::endl;
            Particles.ghost_get<Velocity>();
            div = -Div(V);
            Particles.ghost_get<19>();
            DCPSE_scheme<equations2d1pE, decltype(Particles)> SolverH(Particles, options_solver::LAGRANGE_MULTIPLIER);
            SolverH.impose(Helmholtz, bulk, prop_id<19>());
            SolverH.impose(H, up_p, 0);
            SolverH.impose(H, dw_p, 0);
            tt.start();
            SolverH.solve(H);
            tt.stop();
            std::cout << "Helmholtz Solved in " << tt.getwct() << " seconds." << std::endl;
            Particles.ghost_get<17>();
            V = V + Grad(H);
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<1>(p)[0] = 0;
                Particles.getProp<1>(p)[1] = 0;
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<1>(p)[0] = 0;
                Particles.getProp<1>(p)[1] = 0;
            }
            P = P + Lap(H);
            Particles.ghost_get<Velocity>();
            Particles.ghost_get<Pressure>();
            sum = 0;
            sum1 = 0;
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                sum += (Particles.getProp<18>(p)[0] - Particles.getProp<1>(p)[0]) *
                       (Particles.getProp<18>(p)[0] - Particles.getProp<1>(p)[0]) +
                       (Particles.getProp<18>(p)[1] - Particles.getProp<1>(p)[1]) *
                       (Particles.getProp<18>(p)[1] - Particles.getProp<1>(p)[1]);
                sum1 += Particles.getProp<1>(p)[0] * Particles.getProp<1>(p)[0] +
                        Particles.getProp<1>(p)[1] * Particles.getProp<1>(p)[1];
            }
            sum = sqrt(sum);
            sum1 = sqrt(sum1);
            V_t = V;
            std::cout << "Rel l2 cgs err in V at " << i << "= " << sum / sum1 << std::endl;
            std::cout << "----------------------------------------------------------" << std::endl;
            if (i % 10 == 0)
                Particles.write_frame("Polar", i);
            return;
        }
        Particles.deleteGhost();
        Particles.write_frame("Polar", n + 1);

    }


    BOOST_AUTO_TEST_CASE(Active2DEigen_decouple) {
        timer tt2;
        tt2.start();
        const size_t sz[2] = {41, 41};
        Box<2, double> box({0, 0}, {10, 10});
        double Lx = box.getHigh(0);
        double Ly = box.getHigh(1);
        size_t bc[2] = {NON_PERIODIC, NON_PERIODIC};
        double spacing = box.getHigh(0) / (sz[0] - 1);
        double rCut = 3.1 * spacing;
        double rCut2 = 3.1 * spacing;
        int ord = 2;
        double sampling_factor = 1.9;
        int ord2 = 2;
        double sampling_factor2 = 1.9;
        Ghost<2, double> ghost(rCut);
/*                                          pol                             V         vort                 Ext    Press     strain       stress                      Mfield,   dPol                      dV         RHS                  f1     f2     f3    f4     f5     f6       H               V_t      div   H_t                                                                                      delmu */
        vector_dist<2, double, aggregate<VectorS<2, double>, VectorS<2, double>, double[2][2], VectorS<2, double>, double, double[2][2], double[2][2], VectorS<2, double>, VectorS<2, double>, VectorS<2, double>, VectorS<2, double>, double, double, double, double, double, double, double, VectorS<2, double>, double, double, double[2], double[2], double[2], double[2], double[2], double[2], double, VectorS<2, double>, VectorS<2, double>, VectorS<2, double>, VectorS<2, double>>> Particles(
                0, box, bc, ghost);

        vector_dist<2, double, aggregate<double, double, VectorS<2, double>>> Particles_subset(0, box, bc, ghost);

        auto it = Particles.getGridIterator(sz);
        while (it.isNext()) {
            Particles.add();
            auto key = it.get();
            double x = key.get(0) * it.getSpacing(0);
            Particles.getLastPos()[0] = x;
            double y = key.get(1) * it.getSpacing(1);
            Particles.getLastPos()[1] = y;
            ++it;
        }

        Particles.map();
        Particles.ghost_get<0>();

        openfpm::vector<aggregate<int>> bulk;
        openfpm::vector<aggregate<int>> bulkP;
        openfpm::vector<aggregate<int>> up_p;
        openfpm::vector<aggregate<int>> dw_p;
        openfpm::vector<aggregate<int>> l_p;
        openfpm::vector<aggregate<int>> r_p;
        openfpm::vector<aggregate<int>> ref_p;

        constexpr int x = 0;
        constexpr int y = 1;

        constexpr int Polarization = 0;
        constexpr int Velocity = 1;
        constexpr int Vorticity = 2;
        constexpr int ExtForce = 3;
        constexpr int Pressure = 4;
        constexpr int Strain_rate = 5;
        constexpr int Stress = 6;
        constexpr int MolField = 7;
        auto Pos = getV<PROP_POS>(Particles);

        auto Pol = getV<Polarization>(Particles);
        auto V = getV<Velocity>(Particles);
        auto W = getV<Vorticity>(Particles);
        auto g = getV<ExtForce>(Particles);
        auto P = getV<Pressure>(Particles);
        auto u = getV<Strain_rate>(Particles);
        auto sigma = getV<Stress>(Particles);
        auto h = getV<MolField>(Particles);
        auto dP = getV<8>(Particles);
        auto dV = getV<9>(Particles);
        auto RHS = getV<10>(Particles);
        auto f1 = getV<11>(Particles);
        auto f2 = getV<12>(Particles);
        auto f3 = getV<13>(Particles);
        auto f4 = getV<14>(Particles);
        auto f5 = getV<15>(Particles);
        auto f6 = getV<16>(Particles);
        auto H = getV<17>(Particles);
        auto V_t = getV<18>(Particles);
        auto div = getV<19>(Particles);
        auto H_t = getV<20>(Particles);
        auto Df1 = getV<21>(Particles);
        auto Df2 = getV<22>(Particles);
        auto Df3 = getV<23>(Particles);
        auto Df4 = getV<24>(Particles);
        auto Df5 = getV<25>(Particles);
        auto Df6 = getV<26>(Particles);
        auto delmu = getV<27>(Particles);
        auto k1 = getV<28>(Particles);
        auto k2 = getV<29>(Particles);
        auto k3 = getV<30>(Particles);
        auto k4 = getV<31>(Particles);

        auto P_bulk = getV<0>(Particles_subset); //Pressure only on inside
        auto H_bulk = getV<1>(Particles_subset); //Pressure only on inside
        auto Grad_bulk = getV<2>(Particles_subset);


        double eta = 1.0;
        double nu = -0.5;
        double gama = 0.1;
        double zeta = 0.07;
        double Ks = 1.0;
        double Kb = 1.0;
        double lambda = 0.1;
        //double delmu = -1.0;
        g = 0;
        delmu = -1.0;
        P = 0;
        V = 0;
        P_bulk = 0;
        // Here fill up the boxes for particle boundary detection.

        Box<2, double> up({box.getLow(0) - spacing / 2.0, box.getHigh(1) - spacing / 2.0},
                          {box.getHigh(0) + spacing / 2.0, box.getHigh(1) + spacing / 2.0});

        Box<2, double> down({box.getLow(0) - spacing / 2.0, box.getLow(1) - spacing / 2.0},
                            {box.getHigh(0) + spacing / 2.0, box.getLow(1) + spacing / 2.0});

        Box<2, double> left({box.getLow(0) - spacing / 2.0, box.getLow(1) + spacing / 2.0},
                            {box.getLow(0) + spacing / 2.0, box.getHigh(1) - spacing / 2.0});

        Box<2, double> right({box.getHigh(0) - spacing / 2.0, box.getLow(1) + spacing / 2.0},
                             {box.getHigh(0) + spacing / 2.0, box.getHigh(1) - spacing / 2.0});
        Box<2, double> mid({box.getHigh(0) / 2.0 - 0.75 * spacing, box.getHigh(1) / 2.0 - 0.75 * spacing},
                           {box.getHigh(0) / 2.0 + 0.75 * spacing, box.getHigh(1) / 2.0 + 0.75 * spacing});

/*
        Box<2, double> mid({box.getLow(0)/ + 3.1 * spacing, box.getLow(1) + 3.1 * spacing},
                           {box.getHigh(0) - 3.1 * spacing, box.getHigh(1) - 3.1 * spacing});
*/


        openfpm::vector<Box<2, double>> boxes;
        boxes.add(up);
        boxes.add(down);
        boxes.add(left);
        boxes.add(right);
        boxes.add(mid);

        VTKWriter<openfpm::vector<Box<2, double>>, VECTOR_BOX> vtk_box;
        vtk_box.add(boxes);
        vtk_box.write("vtk_box.vtk");

        auto it2 = Particles.getDomainIterator();

        while (it2.isNext()) {
            auto p = it2.get();
            Point<2, double> xp = Particles.getPos(p);
            Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * xp[x] - Lx) / Lx) - sin((2 * xp[y] - Ly) / Ly)));
            Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * xp[x] - Lx) / Lx) - sin((2 * xp[y] - Ly) / Ly)));
            if (up.isInside(xp) == true) {
                up_p.add();
                up_p.last().get<0>() = p.getKey();
            } else if (down.isInside(xp) == true) {
                dw_p.add();
                dw_p.last().get<0>() = p.getKey();
            } else if (left.isInside(xp) == true) {
                l_p.add();
                l_p.last().get<0>() = p.getKey();
            } else if (right.isInside(xp) == true) {
                r_p.add();
                r_p.last().get<0>() = p.getKey();
            } else {
                if (mid.isInside(xp) == true) {
                    ref_p.add();
                    ref_p.last().get<0>() = p.getKey();
                    Particles.getProp<4>(p) = 0;
                } else {
                    bulkP.add();
                    bulkP.last().get<0>() = p.getKey();
                }
                bulk.add();
                bulk.last().get<0>() = p.getKey();
            }


            ++it2;
        }
        for (int i = 0; i < bulk.size(); i++) {
            Particles_subset.add();
            Particles_subset.getLastPos()[0] = Particles.getPos(bulk.template get<0>(i))[0];
            Particles_subset.getLastPos()[1] = Particles.getPos(bulk.template get<0>(i))[1];
        }

        Derivative_x Dx(Particles, ord, rCut, sampling_factor, support_options::RADIUS), Bulk_Dx(Particles_subset, ord2,
                                                                                                 rCut2,
                                                                                                 sampling_factor2,
                                                                                                 support_options::RADIUS);
        Derivative_y Dy(Particles, ord, rCut, sampling_factor, support_options::RADIUS), Bulk_Dy(Particles_subset, ord2,
                                                                                                 rCut2,
                                                                                                 sampling_factor2,
                                                                                                 support_options::RADIUS);
        Derivative_xy Dxy(Particles, ord, rCut, sampling_factor, support_options::RADIUS);
        auto Dyx = Dxy;
        Derivative_xx Dxx(Particles, ord, rCut, sampling_factor, support_options::RADIUS), Bulk_Dxx(Particles_subset,
                                                                                                    ord2, rCut2,
                                                                                                    sampling_factor2,
                                                                                                    support_options::RADIUS);
        Derivative_yy Dyy(Particles, ord, rCut, sampling_factor, support_options::RADIUS), Bulk_Dyy(Particles_subset,
                                                                                                    ord2, rCut2,
                                                                                                    sampling_factor2,
                                                                                                    support_options::RADIUS);

        petsc_solver<double> solverPetsc;
        solverPetsc.setSolver(KSPGMRES);
        //solverPetsc.setRestart(250);
        //solverPetsc.setPreconditioner(PCJACOBI);
        petsc_solver<double> solverPetsc2;
        solverPetsc2.setSolver(KSPGMRES);
        //solverPetsc2.setRestart(250);
        /*solverPetsc2.setPreconditioner(PCJACOBI);
        solverPetsc2.setRelTol(1e-6);
        solverPetsc2.setAbsTol(1e-5);
        solverPetsc2.setDivTol(1e4);*/
        eq_id vx, vy, ic;
        timer tt;
        timer tt3;
        vx.setId(0);
        vy.setId(1);
        double V_err_eps = 1e-4;
        double V_err = 1;
        int n = 0;
        int ctr = 0;
        double dt = 3e-3;
        double tim = 0;
        double tf = 3e-3;
        while (tim < tf) {
            tt.start();
            Particles.ghost_get<Polarization>();
            sigma[x][x] =
                    -Ks * Dx(Pol[x]) * Dx(Pol[x]) - Kb * Dx(Pol[y]) * Dx(Pol[y]) + (Kb - Ks) * Dy(Pol[x]) * Dx(Pol[y]);
            sigma[x][y] =
                    -Ks * Dy(Pol[y]) * Dx(Pol[y]) - Kb * Dy(Pol[x]) * Dx(Pol[x]) + (Kb - Ks) * Dx(Pol[y]) * Dx(Pol[x]);
            sigma[y][x] =
                    -Ks * Dx(Pol[x]) * Dy(Pol[x]) - Kb * Dx(Pol[y]) * Dy(Pol[y]) + (Kb - Ks) * Dy(Pol[x]) * Dy(Pol[y]);
            sigma[y][y] =
                    -Ks * Dy(Pol[y]) * Dy(Pol[y]) - Kb * Dy(Pol[x]) * Dy(Pol[x]) + (Kb - Ks) * Dx(Pol[y]) * Dy(Pol[x]);
            Particles.ghost_get<Stress>();


            h[y] = (Pol[x] * (Ks * Dyy(Pol[y]) + Kb * Dxx(Pol[y]) + (Ks - Kb) * Dxy(Pol[x])) -
                    Pol[y] * (Ks * Dxx(Pol[x]) + Kb * Dyy(Pol[x]) + (Ks - Kb) * Dxy(Pol[y])));
            Particles.ghost_get<MolField>();

            f1 = gama * nu * Pol[x] * Pol[x] * (Pol[x] * Pol[x] - Pol[y] * Pol[y]) /
                 (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            f2 = 2.0 * gama * nu * Pol[x] * Pol[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y]) /
                 (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            f3 = gama * nu * Pol[y] * Pol[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y]) /
                 (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            f4 = 2.0 * gama * nu * Pol[x] * Pol[x] * Pol[x] * Pol[y] / (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            f5 = 4.0 * gama * nu * Pol[x] * Pol[x] * Pol[y] * Pol[y] / (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            f6 = 2.0 * gama * nu * Pol[x] * Pol[y] * Pol[y] * Pol[y] / (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            Particles.ghost_get<11, 12, 13, 14, 15, 16>();
            Df1[x] = Dx(f1);
            Df2[x] = Dx(f2);
            Df3[x] = Dx(f3);
            Df4[x] = Dx(f4);
            Df5[x] = Dx(f5);
            Df6[x] = Dx(f6);

            Df1[y] = Dy(f1);
            Df2[y] = Dy(f2);
            Df3[y] = Dy(f3);
            Df4[y] = Dy(f4);
            Df5[y] = Dy(f5);
            Df6[y] = Dy(f6);
            Particles.ghost_get<21, 22, 23, 24, 25, 26>();


            dV[x] = -0.5 * Dy(h[y]) + zeta * Dx(delmu * Pol[x] * Pol[x]) + zeta * Dy(delmu * Pol[x] * Pol[y]) -
                    zeta * Dx(0.5 * delmu * (Pol[x] * Pol[x] + Pol[y] * Pol[y])) -
                    0.5 * nu * Dx(-2.0 * h[y] * Pol[x] * Pol[y])
                    - 0.5 * nu * Dy(h[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y])) - Dx(sigma[x][x]) - Dy(sigma[x][y]) -
                    g[x]
                    - 0.5 * nu * Dx(-gama * lambda * delmu * (Pol[x] * Pol[x] - Pol[y] * Pol[y]))
                    - 0.5 * Dy(-2.0 * gama * lambda * delmu * (Pol[x] * Pol[y]));


            dV[y] = -0.5 * Dx(-h[y]) + zeta * Dy(delmu * Pol[y] * Pol[y]) + zeta * Dx(delmu * Pol[x] * Pol[y]) -
                    zeta * Dy(0.5 * delmu * (Pol[x] * Pol[x] + Pol[y] * Pol[y])) -
                    0.5 * nu * Dy(2.0 * h[y] * Pol[x] * Pol[y])
                    - 0.5 * nu * Dx(h[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y])) - Dx(sigma[y][x]) - Dy(sigma[y][y]) -
                    g[y]
                    - 0.5 * nu * Dy(gama * lambda * delmu * (Pol[x] * Pol[x] - Pol[y] * Pol[y]))
                    - 0.5 * Dx(-2.0 * gama * lambda * delmu * (Pol[x] * Pol[y]));
            Particles.ghost_get<9>();
            //Velocity Solution
            auto Stokes1 = eta * (Dxx(V[x]) + Dyy(V[x]))
                           + 0.5 * nu * (Df1[x] * Dx(V[x]) + f1 * Dxx(V[x]))
                           + 0.5 * nu * (Df2[x] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f2 * 0.5 * (Dxx(V[y]) + Dyx(V[x])))
                           + 0.5 * nu * (Df3[x] * Dy(V[y]) + f3 * Dyx(V[y]))
                           + 0.5 * nu * (Df4[y] * Dx(V[x]) + f4 * Dxy(V[x]))
                           + 0.5 * nu * (Df5[y] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f5 * 0.5 * (Dxy(V[y]) + Dyy(V[x])))
                           + 0.5 * nu * (Df6[y] * Dy(V[y]) + f6 * Dyy(V[y]));
            auto Stokes2 = eta * (Dxx(V[y]) + Dyy(V[y]))
                           - 0.5 * nu * (Df1[y] * Dx(V[x]) + f1 * Dxy(V[x]))
                           - 0.5 * nu * (Df2[y] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f2 * 0.5 * (Dxy(V[y]) + Dyy(V[x])))
                           - 0.5 * nu * (Df3[y] * Dy(V[y]) + f3 * Dyy(V[y]))
                           + 0.5 * nu * (Df4[x] * Dx(V[x]) + f4 * Dxx(V[x]))
                           + 0.5 * nu * (Df5[x] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f5 * 0.5 * (Dxx(V[y]) + Dyx(V[x])))
                           + 0.5 * nu * (Df6[x] * Dy(V[y]) + f6 * Dyx(V[y]));
            tt.stop();
            std::cout << "Init of Velocity took " << tt.getwct() << " seconds." << std::endl;
            tt.start();
            //auto Pressure_Poisson = Bulk_Dxx(P_bulk) + Bulk_Dyy(P_bulk);
            auto Pressure_Poisson = Dxx(P) + Dyy(P);
            div = -(Dx(dV[x]) + Dy(dV[y]));
            /*for (int i = 0 ; i < bulk.size() ; i++) {
                Particles_subset.getProp<1>(i) = Particles.template getProp<19>(bulk.template get<0>(i));
            }*/
            Particles.ghost_get<19>();
            DCPSE_scheme<equations2d1E, decltype(Particles)> SolverH(Particles);
            SolverH.impose(Pressure_Poisson, bulkP, prop_id<19>());
            SolverH.impose(Dy(P), up_p, 0);
            SolverH.impose(Dy(P), dw_p, 0);
            SolverH.impose(Dx(P), l_p, 0);
            SolverH.impose(Dx(P), r_p, 0);
            SolverH.impose(P, ref_p, 0);
            SolverH.solve(P);
            std::cout << "Pressure Poisson Solved in " << tt.getwct() << " seconds." << std::endl;
            Particles.ghost_get<Pressure>();
            for (int i = 0; i < bulk.size(); i++) {
                Particles_subset.getProp<0>(i) = Particles.template getProp<4>(bulk.template get<0>(i));
            }
            RHS[x] = dV[x];
            RHS[y] = dV[y];
            Grad_bulk[x] = Bulk_Dx(P_bulk);
            Grad_bulk[y] = Bulk_Dy(P_bulk);
            for (int i = 0; i < bulk.size(); i++) {
                Particles.template getProp<10>(bulk.template get<0>(i))[x] += Particles_subset.getProp<2>(i)[x];
                Particles.template getProp<10>(bulk.template get<0>(i))[y] += Particles_subset.getProp<2>(i)[y];
            }
            Particles.ghost_get<10>();
            DCPSE_scheme<equations2d2E, decltype(Particles)> Solver(Particles);
            Solver.impose(Stokes1, bulk, RHS[0], vx);
            Solver.impose(Stokes2, bulk, RHS[1], vy);
            Solver.impose(V[x], up_p, 0, vx);
            Solver.impose(V[y], up_p, 0, vy);
            Solver.impose(V[x], dw_p, 0, vx);
            Solver.impose(V[y], dw_p, 0, vy);
            Solver.impose(V[x], l_p, 0, vx);
            Solver.impose(V[y], l_p, 0, vy);
            Solver.impose(V[x], r_p, 0, vx);
            Solver.impose(V[y], r_p, 0, vy);
            tt.start();
            Solver.solve(V[x], V[y]);
            tt.stop();
            for (int i = 0; i < bulk.size(); i++) {
                Particles.template getProp<4>(bulk.template get<0>(i)) = Particles_subset.getProp<0>(i);
            }

            std::cout << "Stokes Solved in " << tt.getwct() << " seconds." << std::endl;
            std::cout << "----------------------------------------------------------" << std::endl;
            //Particles.write("Polar_decouple");
            return;
            tim += dt;
        }

    }
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    BOOST_AUTO_TEST_CASE(Active2DEigen_saddle) {
        timer tt2;
        tt2.start();
        const size_t sz[2] = {31,31};
        Box<2, double> box({0, 0}, {10, 10});
        double Lx = box.getHigh(0);
        double Ly = box.getHigh(1);
        size_t bc[2] = {NON_PERIODIC, NON_PERIODIC};
        double spacing = box.getHigh(0) / (sz[0] - 1);
        double rCut = 3.1 * spacing;
        double rCut2 = 3.1 * spacing;
        int ord = 2;
        int ord2 = 2;
        double sampling_factor = 3.4;
        double sampling_factor2 = 1.6;
        double alpha_V = 1.0;
        double alpha_P = 1.0;
        Ghost<2, double> ghost(rCut);

        auto &v_cl = create_vcluster();

/*                                          pol                             V         vort                 Ext    Press     strain       stress                      Mfield,   dPol                      dV         RHS                  f1     f2     f3    f4     f5     f6       H               V_t      div    H_t                                                                                      delmu */
        vector_dist<2, double, aggregate<VectorS<2, double>, VectorS<2, double>, double[2][2], VectorS<2, double>, double, double[2][2], double[2][2], VectorS<2, double>, VectorS<2, double>, VectorS<2, double>, VectorS<2, double>, double, double, double, double, double, double, double, VectorS<2, double>, double, double, double[2], double[2], double[2], double[2], double[2], double[2], double, VectorS<2, double>, VectorS<2, double>, VectorS<2, double>, VectorS<2, double>, double,double,double>> Particles(
                0, box, bc, ghost);
        vector_dist<2, double, aggregate<double, double, VectorS<2, double>>> Particles_subset(
                Particles.getDecomposition(), 0);

        auto it = Particles.getGridIterator(sz);
        while (it.isNext()) {
            Particles.add();
            auto key = it.get();
            double x = key.get(0) * it.getSpacing(0);
            Particles.getLastPos()[0] = x;
            double y = key.get(1) * it.getSpacing(1);
            Particles.getLastPos()[1] = y;
            ++it;
        }

        Particles.map();
        Particles.ghost_get<0>();

        openfpm::vector<aggregate<int>> bulk;
        openfpm::vector<aggregate<int>> bulk_P;
        openfpm::vector<aggregate<int>> ref_p;
        openfpm::vector<aggregate<int>> up_p;
        openfpm::vector<aggregate<int>> dw_p;
        openfpm::vector<aggregate<int>> l_p;
        openfpm::vector<aggregate<int>> r_p;
        openfpm::vector<aggregate<int>> up_p1;
        openfpm::vector<aggregate<int>> dw_p1;
        openfpm::vector<aggregate<int>> l_p1;
        openfpm::vector<aggregate<int>> r_p1;
        openfpm::vector<aggregate<int>> corner_ul;
        openfpm::vector<aggregate<int>> corner_ur;
        openfpm::vector<aggregate<int>> corner_dl;
        openfpm::vector<aggregate<int>> corner_dr;


        constexpr int x = 0;
        constexpr int y = 1;

        constexpr int Polarization = 0;
        constexpr int Velocity = 1;
        constexpr int Vorticity = 2;
        constexpr int ExtForce = 3;
        constexpr int Pressure = 4;
        constexpr int Strain_rate = 5;
        constexpr int Stress = 6;
        constexpr int MolField = 7;
        auto Pos = getV<PROP_POS>(Particles);

        auto Pol = getV<Polarization>(Particles);
        auto V = getV<Velocity>(Particles);
        auto W = getV<Vorticity>(Particles);
        auto g = getV<ExtForce>(Particles);
        auto P = getV<Pressure>(Particles);
        auto P_bulk = getV<0>(Particles_subset); //Pressure only on inside
        auto u = getV<Strain_rate>(Particles);
        auto sigma = getV<Stress>(Particles);
        auto h = getV<MolField>(Particles);
        auto dPol = getV<8>(Particles);
        auto dV = getV<9>(Particles);
        auto RHS = getV<10>(Particles);
        auto f1 = getV<11>(Particles);
        auto f2 = getV<12>(Particles);
        auto f3 = getV<13>(Particles);
        auto f4 = getV<14>(Particles);
        auto f5 = getV<15>(Particles);
        auto f6 = getV<16>(Particles);
        auto H = getV<17>(Particles);
        auto H_bulk = getV<1>(Particles_subset); //Pressure only on inside
        auto Grad_bulk = getV<2>(Particles_subset);

        auto V_t = getV<18>(Particles);
        auto div = getV<19>(Particles);
        auto H_t = getV<20>(Particles);
        auto Df1 = getV<21>(Particles);
        auto Df2 = getV<22>(Particles);
        auto Df3 = getV<23>(Particles);
        auto Df4 = getV<24>(Particles);
        auto Df5 = getV<25>(Particles);
        auto Df6 = getV<26>(Particles);
        auto delmu = getV<27>(Particles);
        auto k1 = getV<28>(Particles);
        auto k2 = getV<29>(Particles);
        auto k3 = getV<30>(Particles);
        auto k4 = getV<31>(Particles);
        auto H_p_b = getV<32>(Particles);
        auto FranckEnergyDensity = getV<33>(Particles);
        auto r = getV<34>(Particles);


        double eta = 1.0;
        double nu = -0.5;
        double gama = 0.1;
        double zeta = 0.07;
        double Ks = 1.0;
        double Kb = 1.0;
        double lambda = 0.1;
        //double delmu = -1.0;
        g = 0;
        delmu = -1.0;
        P = 0;
        P_bulk = 0;
        V = 0;
        // Here fill up the boxes for particle boundary detection.
        Particles.ghost_get<ExtForce,28>();


        Box<2, double> up({box.getLow(0) - spacing / 2.0, box.getHigh(1) - spacing / 2.0},
                          {box.getHigh(0) + spacing / 2.0, box.getHigh(1) + spacing / 2.0});

        Box<2, double> down({box.getLow(0) - spacing / 2.0, box.getLow(1) - spacing / 2.0},
                            {box.getHigh(0) + spacing / 2.0, box.getLow(1) + spacing / 2.0});

        Box<2, double> left({box.getLow(0) - spacing / 2.0, box.getLow(1) - spacing / 2.0},
                            {box.getLow(0) + spacing / 2.0, box.getHigh(1) + spacing / 2.0});

        Box<2, double> right({box.getHigh(0) - spacing / 2.0, box.getLow(1) - spacing / 2.0},
                             {box.getHigh(0) + spacing / 2.0, box.getHigh(1) + spacing / 2.0});
        Box<2, double> mid({box.getHigh(0) / 2.0 - spacing, box.getHigh(1) / 2.0 - spacing / 2.0},
                           {box.getHigh(0) / 2.0, box.getHigh(1) / 2.0 + spacing / 2.0});


        openfpm::vector<Box<2, double>> boxes;
        boxes.add(up);
        boxes.add(down);
        boxes.add(left);
        boxes.add(right);
        boxes.add(mid);

        VTKWriter<openfpm::vector<Box<2, double>>, VECTOR_BOX> vtk_box;
        vtk_box.add(boxes);
        vtk_box.write("vtk_box.vtk");

        auto it2 = Particles.getDomainIterator();
        while (it2.isNext()) {
            auto p = it2.get();
            Point<2, double> xp = Particles.getPos(p);
            Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * xp[x] - Lx) / Lx) - sin((2 * xp[y] - Ly) / Ly)));
            Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * xp[x] - Lx) / Lx) - sin((2 * xp[y] - Ly) / Ly)));
            if (up.isInside(xp) == true) {
                if(left.isInside(xp) == true)
                {
                    corner_ul.add();
                    corner_ul.last().get<0>() = p.getKey();
                }
                else if (right.isInside(xp) == true)
                {
                    corner_ur.add();
                    corner_ur.last().get<0>() = p.getKey();
                }
                else
                {
                    up_p1.add();
                    up_p1.last().get<0>() = p.getKey();
                }
                up_p.add();
                up_p.last().get<0>() = p.getKey();

            } else if (down.isInside(xp) == true) {
                if(left.isInside(xp) == true)
                {
                    corner_dl.add();
                    corner_dl.last().get<0>() = p.getKey();
                }
                else if (right.isInside(xp) == true)
                {
                    corner_dr.add();
                    corner_dr.last().get<0>() = p.getKey();
                }
                else
                {
                    dw_p1.add();
                    dw_p1.last().get<0>() = p.getKey();
                }
                dw_p.add();
                dw_p.last().get<0>() = p.getKey();
            } else if (left.isInside(xp) == true) {
                if(up.isInside(xp) == false && down.isInside(xp) == false) {
                    l_p1.add();
                    l_p1.last().get<0>() = p.getKey();
                }
                l_p.add();
                l_p.last().get<0>() = p.getKey();
            } else if (right.isInside(xp) == true) {
                if(up.isInside(xp) == false && down.isInside(xp) == false) {
                    r_p1.add();
                    r_p1.last().get<0>() = p.getKey();
                }
                r_p.add();
                r_p.last().get<0>() = p.getKey();
            }
            else {
                if (mid.isInside(xp) == true)
                {
                    ref_p.add();
                    ref_p.last().get<0>() = p.getKey();
                }
                else{
                    bulk_P.add();
                    bulk_P.last().get<0>() = p.getKey();

                }
                bulk.add();
                bulk.last().get<0>() = p.getKey();
            }
            ++it2;
        }


        for (int i = 0; i < bulk.size(); i++) {
            Particles_subset.add();
            Particles_subset.getLastPos()[0] = Particles.getPos(bulk.template get<0>(i))[0];
            Particles_subset.getLastPos()[1] = Particles.getPos(bulk.template get<0>(i))[1];
        }


        Particles_subset.map();
        Particles_subset.ghost_get<0>();

        Derivative_x Dx(Particles, ord, rCut, sampling_factor, support_options::RADIUS), Bulk_Dx(Particles_subset, ord,
                                                                                                 rCut,
                                                                                                 sampling_factor,
                                                                                                 support_options::RADIUS);
        Derivative_y Dy(Particles, ord, rCut, sampling_factor, support_options::RADIUS), Bulk_Dy(Particles_subset, ord,
                                                                                                 rCut,
                                                                                                 sampling_factor,
                                                                                                 support_options::RADIUS);
        Derivative_xy Dxy(Particles, ord, rCut2, sampling_factor2, support_options::RADIUS);
        auto Dyx = Dxy;
        Derivative_xx Dxx(Particles, ord, rCut2, sampling_factor2,
                          support_options::RADIUS);//, Dxx2(Particles, ord2, rCut2, sampling_factor2, support_options::RADIUS);
        Derivative_yy Dyy(Particles, ord, rCut2, sampling_factor2,
                          support_options::RADIUS);//, Dyy2(Particles, ord2, rCut2, sampling_factor2, support_options::RADIUS);


        V.setVarId(0);
        P.setVarId(2);
        eq_id vx, vy, ic;
        timer tt;
        timer tt3;
        vx.setId(0);
        vy.setId(1);
        ic.setId(2);
        int ctr = 0;
        double dt = 2e-3    ;
        double tim = 0;
        double tf = 2e-1;
        div = 0;
        double sum, sum1, sum_k;
        while (tim <= tf) {
            tt.start();
            Particles.ghost_get<Polarization>();
            sigma[x][x] =
                    -Ks * Dx(Pol[x]) * Dx(Pol[x]) - Kb * Dx(Pol[y]) * Dx(Pol[y]) + (Kb - Ks) * Dy(Pol[x]) * Dx(Pol[y]);
            sigma[x][y] =
                    -Ks * Dy(Pol[y]) * Dx(Pol[y]) - Kb * Dy(Pol[x]) * Dx(Pol[x]) + (Kb - Ks) * Dx(Pol[y]) * Dx(Pol[x]);
            sigma[y][x] =
                    -Ks * Dx(Pol[x]) * Dy(Pol[x]) - Kb * Dx(Pol[y]) * Dy(Pol[y]) + (Kb - Ks) * Dy(Pol[x]) * Dy(Pol[y]);
            sigma[y][y] =
                    -Ks * Dy(Pol[y]) * Dy(Pol[y]) - Kb * Dy(Pol[x]) * Dy(Pol[x]) + (Kb - Ks) * Dx(Pol[y]) * Dy(Pol[x]);
            Particles.ghost_get<Stress>();


            h[y] = (Pol[x] * (Ks * Dyy(Pol[y]) + Kb * Dxx(Pol[y]) + (Ks - Kb) * Dxy(Pol[x])) -
                    Pol[y] * (Ks * Dxx(Pol[x]) + Kb * Dyy(Pol[x]) + (Ks - Kb) * Dxy(Pol[y])));

            Particles.ghost_get<MolField>();

            FranckEnergyDensity = (Ks / 2.0) * ((Dx(Pol[x]) * Dx(Pol[x])) + (Dy(Pol[x]) * Dy(Pol[x])) + (Dx(Pol[y]) * Dx(Pol[y])) + (Dy(Pol[y]) * Dy(Pol[y]))) + ((Kb - Ks) / 2.0) * ((Dx(Pol[y]) - Dy(Pol[x])) * (Dx(Pol[y]) - Dy(Pol[x])));
            Particles.ghost_get<33>();


            f1 = gama * nu * Pol[x] * Pol[x] * (Pol[x] * Pol[x] - Pol[y] * Pol[y]) /
                 (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            f2 = 2.0 * gama * nu * Pol[x] * Pol[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y]) /
                 (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            f3 = gama * nu * Pol[y] * Pol[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y]) /
                 (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            f4 = 2.0 * gama * nu * Pol[x] * Pol[x] * Pol[x] * Pol[y] / (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            f5 = 4.0 * gama * nu * Pol[x] * Pol[x] * Pol[y] * Pol[y] / (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            f6 = 2.0 * gama * nu * Pol[x] * Pol[y] * Pol[y] * Pol[y] / (Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            Particles.ghost_get<11, 12, 13, 14, 15, 16>();
            Df1[x] = Dx(f1);
            Df2[x] = Dx(f2);
            Df3[x] = Dx(f3);
            Df4[x] = Dx(f4);
            Df5[x] = Dx(f5);
            Df6[x] = Dx(f6);

            Df1[y] = Dy(f1);
            Df2[y] = Dy(f2);
            Df3[y] = Dy(f3);
            Df4[y] = Dy(f4);
            Df5[y] = Dy(f5);
            Df6[y] = Dy(f6);
            Particles.ghost_get<21, 22, 23, 24, 25, 26>();


            dV[x] = -0.5 * Dy(h[y]) + zeta * Dx(delmu * Pol[x] * Pol[x]) + zeta * Dy(delmu * Pol[x] * Pol[y]) -
                    zeta * Dx(0.5 * delmu * (Pol[x] * Pol[x] + Pol[y] * Pol[y])) -
                    0.5 * nu * Dx(-2.0 * h[y] * Pol[x] * Pol[y])
                    - 0.5 * nu * Dy(h[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y])) - Dx(sigma[x][x]) - Dy(sigma[x][y]) -
                    g[x]
                    - 0.5 * nu * Dx(-gama * lambda * delmu * (Pol[x] * Pol[x] - Pol[y] * Pol[y]))
                    - 0.5 * Dy(-2.0 * gama * lambda * delmu * (Pol[x] * Pol[y]));


            dV[y] = -0.5 * Dx(-h[y]) + zeta * Dy(delmu * Pol[y] * Pol[y]) + zeta * Dx(delmu * Pol[x] * Pol[y]) -
                    zeta * Dy(0.5 * delmu * (Pol[x] * Pol[x] + Pol[y] * Pol[y])) -
                    0.5 * nu * Dy(2.0 * h[y] * Pol[x] * Pol[y])
                    - 0.5 * nu * Dx(h[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y])) - Dx(sigma[y][x]) - Dy(sigma[y][y]) -
                    g[y]
                    - 0.5 * nu * Dy(gama * lambda * delmu * (Pol[x] * Pol[x] - Pol[y] * Pol[y]))
                    - 0.5 * Dx(-2.0 * gama * lambda * delmu * (Pol[x] * Pol[y]));
            Particles.ghost_get<9>();


            //Particles.write("PolarI");
            //Velocity Solution n iterations
            auto Stokes1 = eta * (Dxx(V[x]) + Dyy(V[x])) - Dx(P)
                           + 0.5 * nu * (Df1[x] * Dx(V[x]) + f1 * Dxx(V[x]))
                           + 0.5 * nu * (Df2[x] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f2 * 0.5 * (Dxx(V[y]) + Dyx(V[x])))
                           + 0.5 * nu * (Df3[x] * Dy(V[y]) + f3 * Dyx(V[y]))
                           + 0.5 * nu * (Df4[y] * Dx(V[x]) + f4 * Dxy(V[x]))
                           + 0.5 * nu * (Df5[y] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f5 * 0.5 * (Dxy(V[y]) + Dyy(V[x])))
                           + 0.5 * nu * (Df6[y] * Dy(V[y]) + f6 * Dyy(V[y]));
            auto Stokes2 = eta * (Dxx(V[y]) + Dyy(V[y])) - Dy(P)
                           - 0.5 * nu * (Df1[y] * Dx(V[x]) + f1 * Dxy(V[x]))
                           - 0.5 * nu * (Df2[y] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f2 * 0.5 * (Dxy(V[y]) + Dyy(V[x])))
                           - 0.5 * nu * (Df3[y] * Dy(V[y]) + f3 * Dyy(V[y]))
                           + 0.5 * nu * (Df4[x] * Dx(V[x]) + f4 * Dxx(V[x]))
                           + 0.5 * nu * (Df5[x] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f5 * 0.5 * (Dxx(V[y]) + Dyx(V[x])))
                           + 0.5 * nu * (Df6[x] * Dy(V[y]) + f6 * Dyx(V[y]));
            auto continuity = (Dx(V[x]) + Dy(V[y]));
            tt.stop();
            std::cout << "Init of Velocity took " << tt.getwct() << " seconds." << std::endl;
            tt.start();
            RHS[x] = dV[x];
            RHS[y] = dV[y];
            Particles.ghost_get<19>();
            DCPSE_scheme<equations2d3E, decltype(Particles)> Solver(Particles);
            Solver.impose(Stokes1, bulk, RHS[0], vx);
            Solver.impose(Stokes2, bulk, RHS[1], vy);
            Solver.impose(continuity, bulk_P, 0, ic);
            Solver.impose(V[x], up_p, 0, vx);
            Solver.impose(V[y], up_p, 0, vy);
            Solver.impose(V[x], dw_p, 0, vx);
            Solver.impose(V[y], dw_p, 0, vy);
            Solver.impose(V[x], l_p, 0, vx);
            Solver.impose(V[y], l_p, 0, vy);
            Solver.impose(V[x], r_p, 0, vx);
            Solver.impose(V[y], r_p, 0, vy);
            Solver.impose(P, ref_p, 0, ic);
/*            Solver.impose(P, up_p, 0, ic);
            Solver.impose(P, dw_p, 0, ic);
            Solver.impose(P, l_p, 0, ic);
            Solver.impose(P, r_p, 0, ic);*/


            Solver.impose(Dx(P), up_p1, 0, ic);
            Solver.impose(-Dy(P), dw_p1, 0, ic);
            Solver.impose(-Dx(P), l_p1, 0, ic);
            Solver.impose(Dx(P), r_p1, 0, ic);
            Solver.impose(Dy(P)-Dx(P), corner_ul,0, ic);
            Solver.impose(Dx(P)+Dy(P), corner_ur,0, ic);
            Solver.impose(-Dy(P)-Dx(P), corner_dl,0, ic);
            Solver.impose(Dx(P)-Dy(P), corner_dr,0, ic);
            tt.start();
            BOOST_TEST_MESSAGE("Solver Imposed...");
            Solver.solve(V[x], V[y], P);
            BOOST_TEST_MESSAGE("Solver Solved...");
            //Solver.solve(V[x], V[y],P);
            tt.stop();
            std::cout << "Stokes with Pressure Solved in " << tt.getwct() << " seconds." << std::endl;
            std::cout << "----------------------------------------------------------" << std::endl;
            Particles.write_frame("Polar_saddle", 0);
/*            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<1>(p)[0] = 0;
                Particles.getProp<1>(p)[1] = 0;
                Particles.getProp<4>(p) = 0;

            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<1>(p)[0] = 0;
                Particles.getProp<1>(p)[1] = 0;
                Particles.getProp<4>(p) = 0;
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<1>(p)[0] = 0;
                Particles.getProp<1>(p)[1] = 0;
                Particles.getProp<4>(p) = 0;
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<1>(p)[0] = 0;
                Particles.getProp<1>(p)[1] = 0;
                Particles.getProp<4>(p) = 0;
            }*/
            Particles.ghost_get<Velocity>();
            Particles.ghost_get<Pressure>();

            u[x][x] = Dx(V[x]);
            u[x][y] = 0.5 * (Dx(V[y]) + Dy(V[x]));
            u[y][x] = 0.5 * (Dy(V[x]) + Dx(V[y]));
            u[y][y] = Dy(V[y]);

            W[x][x] = 0;
            W[x][y] = 0.5 * (Dy(V[x]) - Dx(V[y]));
            W[y][x] = 0.5 * (Dx(V[y]) - Dy(V[x]));
            W[y][y] = 0;


            h[x] = -gama * (lambda * delmu - nu * (u[x][x] * Pol[x] * Pol[x] +  u[y][y] * Pol[y] * Pol[y] + 2* u[x][y] * Pol[x] * Pol[y]) / (Pol[x] * Pol[x] + Pol[y] * Pol[y]));


            Particles.ghost_get<MolField, Strain_rate, Vorticity>();
            Particles.deleteGhost();
            Particles.write_frame("Polar_3e-3", ctr);
            return;
            Particles.ghost_get<MolField, Strain_rate, Vorticity>();
            ctr++;
            H_p_b = Pol[x] * Pol[x] + Pol[y] * Pol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<32>(p) =(Particles.getProp<32>(p)== 0) ? 1 : Particles.getProp<32>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<32>(p) =(Particles.getProp<32>(p)== 0) ? 1 : Particles.getProp<32>(p);
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<32>(p) =(Particles.getProp<32>(p)== 0) ? 1 : Particles.getProp<32>(p);
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<32>(p) =(Particles.getProp<32>(p)== 0) ? 1 : Particles.getProp<32>(p);
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<32>(p) =(Particles.getProp<32>(p)== 0) ? 1 : Particles.getProp<32>(p);
            }
            H_p_b=sqrt(H_p_b);

            k1[x] = ((h[x] * Pol[x] - h[y] * Pol[y]) / gama + lambda * delmu * Pol[x] -
                     nu * (u[x][x] * Pol[x] + u[x][y] * Pol[y]) + W[x][x] * Pol[x] +
                     W[x][y] * Pol[y]);// - V[x] * Dx(Pol[x]) - V[y] * Dy(Pol[x]));
            k1[y] = ((h[x] * Pol[y] + h[y] * Pol[x]) / gama + lambda * delmu * Pol[y] -
                     nu * (u[y][x] * Pol[x] + u[y][y] * Pol[y]) + W[y][x] * Pol[x] +
                     W[y][y] * Pol[y]);// - V[x] * Dx(Pol[y]) - V[y] * Dy(Pol[y]));
            Particles.ghost_get<28>();

            H_t = H_p_b;//+0.5*dt*(k1[x]*k1[x]+k1[y]*k1[y]);
            dPol = Pol + (0.5 * dt) * k1;
            dPol = dPol / H_t;
            Particles.ghost_get<8>();
            r=dPol[x]*dPol[x]+dPol[y]*dPol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }

            h[y] = (dPol[x] * (Ks * Dyy(dPol[y]) + Kb * Dxx(dPol[y]) + (Ks - Kb) * Dxy(dPol[x])) -
                    dPol[y] * (Ks * Dxx(dPol[x]) + Kb * Dyy(dPol[x]) + (Ks - Kb) * Dxy(dPol[y])));

            h[x] = -gama * (lambda * delmu - nu * ((u[x][x] * dPol[x] * dPol[x] +  u[y][y] * dPol[y] * dPol[y] + 2* u[x][y] * dPol[x] * dPol[y])/(r)));
            Particles.ghost_get<7>();


            k2[x] = ((h[x] * (dPol[x]) - h[y] * (dPol[y])) / gama +
                     lambda * delmu * (dPol[x]) -
                     nu * (u[x][x] * (dPol[x]) + u[x][y] * (dPol[y])) +
                     W[x][x] * (dPol[x]) + W[x][y] * (dPol[y])); //-V[x] * Dx((dPol[x])) - V[y] * Dy((dPol[x])));
            k2[y] = ((h[x] * (dPol[y]) + h[y] * (dPol[x])) / gama +
                     lambda * delmu * (dPol[y]) -
                     nu * (u[y][x] * (dPol[x]) + u[y][y] * (dPol[y])) +
                     W[y][x] * (dPol[x]) + W[y][y] * (dPol[y])); //-V[x] * Dx((dPol[y])) - V[y] * Dy((dPol[y])));
            Particles.ghost_get<29>();
            H_t = H_p_b;//+0.5*dt*(k2[x]*k2[x]+k2[y]*k2[y]);
            dPol = Pol + (0.5 * dt) * k2;
            dPol = dPol / H_t;
            Particles.ghost_get<8>();
            r=dPol[x]*dPol[x]+dPol[y]*dPol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }

            h[y] = (dPol[x] * (Ks * Dyy(dPol[y]) + Kb * Dxx(dPol[y]) + (Ks - Kb) * Dxy(dPol[x])) -
                    dPol[y] * (Ks * Dxx(dPol[x]) + Kb * Dyy(dPol[x]) + (Ks - Kb) * Dxy(dPol[y])));

            h[x] = -gama * (lambda * delmu - nu * ((u[x][x] * dPol[x] * dPol[x] +  u[y][y] * dPol[y] * dPol[y] + 2* u[x][y] * dPol[x] * dPol[y])/(r)));

            k3[x] = ((h[x] * (dPol[x]) - h[y] * (dPol[y])) / gama +
                     lambda * delmu * (dPol[x]) -
                     nu * (u[x][x] * (dPol[x]) + u[x][y] * (dPol[y])) +
                     W[x][x] * (dPol[x]) + W[x][y] * (dPol[y]));
            // -V[x] * Dx((dPol[x])) - V[y] * Dy((dPol[x])));
            k3[y] = ((h[x] * (dPol[y]) + h[y] * (dPol[x])) / gama +
                     lambda * delmu * (dPol[y]) -
                     nu * (u[y][x] * (dPol[x]) + u[y][y] * (dPol[y])) +
                     W[y][x] * (dPol[x]) + W[y][y] * (dPol[y]));
            // -V[x] * Dx((dPol[y])) - V[y] * Dy((dPol[y])));
            Particles.ghost_get<30>();
            H_t =  H_p_b;//+dt*(k3[x]*k3[x]+k3[y]*k3[y]);
            dPol = Pol + (dt * k3);
            dPol = dPol / H_t;
            Particles.ghost_get<8>();
            r=dPol[x]*dPol[x]+dPol[y]*dPol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<34>(p) =(Particles.getProp<34>(p)== 0) ? 1 : Particles.getProp<34>(p);
            }

            h[y] = (dPol[x] * (Ks * Dyy(dPol[y]) + Kb * Dxx(dPol[y]) + (Ks - Kb) * Dxy(dPol[x])) -
                    dPol[y] * (Ks * Dxx(dPol[x]) + Kb * Dyy(dPol[x]) + (Ks - Kb) * Dxy(dPol[y])));

            h[x] = -gama * (lambda * delmu - nu * ((u[x][x] * dPol[x] * dPol[x] +  u[y][y] * dPol[y] * dPol[y] + 2* u[x][y] * dPol[x] * dPol[y])/(r)));
            Particles.ghost_get<7>();


            k4[x] = ((h[x] * (dPol[x]) - h[y] * (dPol[y])) / gama +
                     lambda * delmu * (dPol[x]) -
                     nu * (u[x][x] * (dPol[x]) + u[x][y] * (dPol[y])) +
                     W[x][x] * (dPol[x]) +
                     W[x][y] * (dPol[y]));//   -V[x]*Dx( (dt * k3[x]+Pol[x])) -V[y]*Dy( (dt * k3[x]+Pol[x])));
            k4[y] = ((h[x] * (dPol[y]) + h[y] * (dPol[x])) / gama +
                     lambda * delmu * (dPol[y]) -
                     nu * (u[y][x] * (dPol[x]) + u[y][y] * (dPol[y])) +
                     W[y][x] * (dPol[x]) +
                     W[y][y] * (dPol[y]));//  -V[x]*Dx( (dt * k3[y]+Pol*[y])) -V[y]*Dy( (dt * k3[y]+Pol[y])));
            Particles.ghost_get<31>();

            Pol = Pol + (dt / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4);
            Pol = Pol / H_p_b;

            H_p_b = sqrt(Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            Pol = Pol / H_p_b;

            Pos = Pos + dt * V;
            Particles.map();
            Particles.ghost_get<0>();
            indexUpdate(Particles, Particles_subset, up_p, dw_p, l_p, r_p,up_p1, dw_p1, l_p1, r_p1,corner_ul,corner_ur,corner_dl,corner_dr, bulk, up, down, left, right);
            Particles_subset.map();
            Particles_subset.ghost_get<0>();

            Particles_subset.write("debug");
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }

            Particles.ghost_get<0, Vorticity, MolField>();
            Particles_subset.ghost_get<0,1,2>();

            tt.start();
            Dx.update(Particles);
            Dy.update(Particles);
            Dxy.update(Particles);
            auto Dyx = Dxy;
            Dxx.update(Particles);
            Dyy.update(Particles);

            Bulk_Dx.update(Particles_subset);
            Bulk_Dy.update(Particles_subset);

            tt.stop();
            std::cout << "Updation of operators took " << tt.getwct() << " seconds." << std::endl;

            std::cout << "Time step " << ctr << " : " << tim << " over." << std::endl;
            tim += dt;
            std::cout << "----------------------------------------------------------" << std::endl;
        }
        Particles.deleteGhost();
        tt2.stop();
        std::cout << "The simulation took " << tt2.getwct() << "Seconds.";
    }



    BOOST_AUTO_TEST_CASE(Active2DEigenWorking) {
        timer tt2;
        tt2.start();
        double boxsize = 10;
        const size_t sz[2] = {41, 41};
        Box<2, double> box({0, 0}, {boxsize, boxsize});
        double Lx = box.getHigh(0);
        double Ly = box.getHigh(1);
        size_t bc[2] = {NON_PERIODIC, NON_PERIODIC};
        double spacing = box.getHigh(0) / (sz[0] - 1);
        double rCut = 3.9 * spacing;
        double rCut2 = 3.9 * spacing;
        int ord = 2;
        int ord2 = 2;
        double sampling_factor = 4.0;
        double sampling_factor2 = 2.4;
        double alpha_V = 1.0;
        double alpha_P = 1.0;
        Ghost<2, double> ghost(rCut);

        auto &v_cl = create_vcluster();

/*                                          pol                             V         vort                 Ext    Press     strain       stress                      Mfield,   dPol                      dV         RHS                  f1     f2     f3    f4     f5     f6       H               V_t      div    H_t                                                                                      delmu */
        vector_dist<2, double, aggregate<VectorS<2, double>, VectorS<2, double>, double[2][2], VectorS<2, double>, double, double[2][2], double[2][2], VectorS<2, double>, VectorS<2, double>, VectorS<2, double>, VectorS<2, double>, double, double, double, double, double, double, double, VectorS<2, double>, double, double, double[2], double[2], double[2], double[2], double[2], double[2], double, VectorS<2, double>, VectorS<2, double>, VectorS<2, double>, VectorS<2, double>, double, double, double>> Particles(
                0, box, bc, ghost);
        vector_dist<2, double, aggregate<double, double, VectorS<2, double>>> Particles_subset(
                Particles.getDecomposition(), 0);
        double x0, y0, x1, y1;
        x0 = box.getLow(0);
        y0 = box.getLow(1);
        x1 = box.getHigh(0);
        y1 = box.getHigh(1);

        auto it = Particles.getGridIterator(sz);
        while (it.isNext()) {
            Particles.add();
            auto key = it.get();
            double x = key.get(0) * it.getSpacing(0);
            Particles.getLastPos()[0] = x;
            double y = key.get(1) * it.getSpacing(1);
            Particles.getLastPos()[1] = y;
            ++it;
        }

        BOOST_TEST_MESSAGE("Sync domain across processors...");

        Particles.map();
        Particles.ghost_get<0>();

        //Particles.write("Par");

        openfpm::vector<aggregate<int>> bulk;
        openfpm::vector<aggregate<int>> up_p;
        openfpm::vector<aggregate<int>> dw_p;
        openfpm::vector<aggregate<int>> l_p;
        openfpm::vector<aggregate<int>> r_p;
        openfpm::vector<aggregate<int>> up_p1;
        openfpm::vector<aggregate<int>> dw_p1;
        openfpm::vector<aggregate<int>> l_p1;
        openfpm::vector<aggregate<int>> r_p1;
        openfpm::vector<aggregate<int>> corner_ul;
        openfpm::vector<aggregate<int>> corner_ur;
        openfpm::vector<aggregate<int>> corner_dl;
        openfpm::vector<aggregate<int>> corner_dr;


        constexpr int x = 0;
        constexpr int y = 1;

        constexpr int Polarization = 0;
        constexpr int Velocity = 1;
        constexpr int Vorticity = 2;
        constexpr int ExtForce = 3;
        constexpr int Pressure = 4;
        constexpr int Strain_rate = 5;
        constexpr int Stress = 6;
        constexpr int MolField = 7;
        auto Pos = getV<PROP_POS>(Particles);

        auto Pol = getV<Polarization>(Particles);
        auto Pol_Bulk = getV<Polarization>(Particles);
        auto V = getV<Velocity>(Particles);
        auto W = getV<Vorticity>(Particles);
        auto g = getV<ExtForce>(Particles);
        auto P = getV<Pressure>(Particles);
        auto P_bulk = getV<0>(Particles_subset); //Pressure only on inside
        auto u = getV<Strain_rate>(Particles);
        auto sigma = getV<Stress>(Particles);
        auto h = getV<MolField>(Particles);
        auto dPol = getV<8>(Particles);
        auto dV = getV<9>(Particles);
        auto RHS = getV<10>(Particles);
        auto f1 = getV<11>(Particles);
        auto f2 = getV<12>(Particles);
        auto f3 = getV<13>(Particles);
        auto f4 = getV<14>(Particles);
        auto f5 = getV<15>(Particles);
        auto f6 = getV<16>(Particles);
        auto H = getV<17>(Particles);
        auto H_bulk = getV<1>(Particles_subset); //Pressure only on inside
        auto Grad_bulk = getV<2>(Particles_subset);

        auto V_t = getV<18>(Particles);
        auto div = getV<19>(Particles);
        auto H_t = getV<20>(Particles);
        auto Df1 = getV<21>(Particles);
        auto Df2 = getV<22>(Particles);
        auto Df3 = getV<23>(Particles);
        auto Df4 = getV<24>(Particles);
        auto Df5 = getV<25>(Particles);
        auto Df6 = getV<26>(Particles);
        auto delmu = getV<27>(Particles);
        auto k1 = getV<28>(Particles);
        auto k2 = getV<29>(Particles);
        auto k3 = getV<30>(Particles);
        auto k4 = getV<31>(Particles);
        auto H_p_b = getV<32>(Particles);
        auto FranckEnergyDensity = getV<33>(Particles);
        auto r = getV<34>(Particles);


        double eta = 1.0;
        double nu = -0.5;
        double gama = 0.1;
        double zeta = 0.07;
        double Ks = 1.0;
        double Kb = 1.0;
        double lambda = 0.1;
        //double delmu = -1.0;
        g = 0;
        delmu = -1.0;
        P = 0;
        P_bulk = 0;
        V = 0;
        // Here fill up the boxes for particle boundary detection.
        Particles.ghost_get<ExtForce, 27>(SKIP_LABELLING);


        Box<2, double> up({x0 - spacing / 2.0, y1 - spacing / 2.0},
                          {x1 + spacing / 2.0, y1 + spacing / 2.0});

        Box<2, double> down({x0 - spacing / 2.0, y0 - spacing / 2.0},
                            {x1 + spacing / 2.0, y0 + spacing / 2.0});

        Box<2, double> left({x0 - spacing / 2.0, y0 - spacing / 2.0},
                            {x0 + spacing / 2.0, y1 + spacing / 2.0});

        Box<2, double> right({x1 - spacing / 2.0, y0 - spacing / 2.0},
                             {x1 + spacing / 2.0, y1 + spacing / 2.0});
        /*Box<2, double> mid({box.getHigh(0) / 2.0 - spacing, box.getHigh(1) / 2.0 - spacing / 2.0},
                           {box.getHigh(0) / 2.0, box.getHigh(1) / 2.0 + spacing / 2.0});
*/


        openfpm::vector<Box<2, double>> boxes;
        boxes.add(up);
        boxes.add(down);
        boxes.add(left);
        boxes.add(right);


        VTKWriter<openfpm::vector<Box<2, double>>, VECTOR_BOX> vtk_box;
        vtk_box.add(boxes);
        vtk_box.write("vtk_box.vtk");

        auto it2 = Particles.getDomainIterator();

        while (it2.isNext()) {
            auto p = it2.get();
            Point<2, double> xp = Particles.getPos(p);
            Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * xp[x] - Lx) / Lx) - sin((2 * xp[y] - Ly) / Ly)));
            Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * xp[x] - Lx) / Lx) - sin((2 * xp[y] - Ly) / Ly)));
            if (up.isInside(xp) == true) {
                if (left.isInside(xp) == true) {
                    corner_ul.add();
                    corner_ul.last().get<0>() = p.getKey();
                } else if (right.isInside(xp) == true) {
                    corner_ur.add();
                    corner_ur.last().get<0>() = p.getKey();
                } else {
                    up_p1.add();
                    up_p1.last().get<0>() = p.getKey();
                }
                up_p.add();
                up_p.last().get<0>() = p.getKey();

            } else if (down.isInside(xp) == true) {
                if (left.isInside(xp) == true) {
                    corner_dl.add();
                    corner_dl.last().get<0>() = p.getKey();
                } else if (right.isInside(xp) == true) {
                    corner_dr.add();
                    corner_dr.last().get<0>() = p.getKey();
                } else {
                    dw_p1.add();
                    dw_p1.last().get<0>() = p.getKey();
                }
                dw_p.add();
                dw_p.last().get<0>() = p.getKey();
            } else if (left.isInside(xp) == true) {
                if (up.isInside(xp) == false && down.isInside(xp) == false) {
                    l_p1.add();
                    l_p1.last().get<0>() = p.getKey();
                }
                l_p.add();
                l_p.last().get<0>() = p.getKey();
            } else if (right.isInside(xp) == true) {
                if (up.isInside(xp) == false && down.isInside(xp) == false) {
                    r_p1.add();
                    r_p1.last().get<0>() = p.getKey();
                }
                r_p.add();
                r_p.last().get<0>() = p.getKey();
            } else {
                bulk.add();
                bulk.last().get<0>() = p.getKey();
            }
            ++it2;
        }


        for (int i = 0; i < bulk.size(); i++) {
            Particles_subset.add();
            Particles_subset.getLastPos()[0] = Particles.getPos(bulk.template get<0>(i))[0];
            Particles_subset.getLastPos()[1] = Particles.getPos(bulk.template get<0>(i))[1];
        }


        Particles_subset.map();
        Particles_subset.ghost_get<0>();

        //Particles_subset.write("Pars");
        Derivative_x Dx(Particles, ord, rCut, sampling_factor, support_options::RADIUS), Bulk_Dx(Particles_subset, ord,
                                                                                                 rCut,
                                                                                                 sampling_factor,
                                                                                                 support_options::RADIUS);
        Derivative_y Dy(Particles, ord, rCut, sampling_factor, support_options::RADIUS), Bulk_Dy(Particles_subset, ord,
                                                                                                 rCut,
                                                                                                 sampling_factor,
                                                                                                 support_options::RADIUS);
        Derivative_xy Dxy(Particles, ord, rCut2, sampling_factor2, support_options::RADIUS);
        auto Dyx = Dxy;
        Derivative_xx Dxx(Particles, ord, rCut2, sampling_factor2,
                          support_options::RADIUS);//, Dxx2(Particles, ord2, rCut2, sampling_factor2, support_options::RADIUS);
        Derivative_yy Dyy(Particles, ord, rCut2, sampling_factor2,
                          support_options::RADIUS);//, Dyy2(Particles, ord2, rCut2, sampling_factor2, support_options::RADIUS);

/*        Derivative_x Dx(Particles, ord, rCut, sampling_factor), Bulk_Dx(Particles_subset, ord, rCut, sampling_factor);
        Derivative_y Dy(Particles, ord, rCut, sampling_factor), Bulk_Dy(Particles_subset, ord, rCut, sampling_factor);
        Derivative_xy Dxy(Particles, ord2, rCut, sampling_factor);
        auto Dyx = Dxy;
        Derivative_xx Dxx(Particles, ord2, rCut2, sampling_factor2),Bulk_Dxx(Particles_subset, ord2, rCut2, sampling_factor2);
        Derivative_yy Dyy(Particles, ord2, rCut2, sampling_factor2),Bulk_Dyy(Particles_subset, ord2, rCut2, sampling_factor2);*/




        eq_id vx, vy;
        timer tt;
        timer tt3;
        vx.setId(0);
        vy.setId(1);
        double V_err_eps = 5 * 1e-4;
        double V_err = 1, V_err_old;
        int n = 0;
        int nmax = 300;
        int ctr = 0, errctr, Vreset = 0;
        double dt = 3 * 1e-3;

        double tim = 0;
        double tf = 1.25;
        div = 0;
        double sum, sum1, sum_k;
        while (tim <= tf) {
            tt.start();
            petsc_solver<double> solverPetsc;
            solverPetsc.setSolver(KSPGMRES);
            //solverPetsc.setRestart(250);
            solverPetsc.setPreconditioner(PCJACOBI);
            petsc_solver<double> solverPetsc2;
            solverPetsc2.setSolver(KSPGMRES);
            solverPetsc2.setPreconditioner(PCJACOBI);

            Particles.ghost_get<Polarization>(SKIP_LABELLING);
            sigma[x][x] =
                    -Ks * Dx(Pol[x]) * Dx(Pol[x]) - Kb * Dx(Pol[y]) * Dx(Pol[y]) + (Kb - Ks) * Dy(Pol[x]) * Dx(Pol[y]);
            sigma[x][y] =
                    -Ks * Dy(Pol[y]) * Dx(Pol[y]) - Kb * Dy(Pol[x]) * Dx(Pol[x]) + (Kb - Ks) * Dx(Pol[y]) * Dx(Pol[x]);
            sigma[y][x] =
                    -Ks * Dx(Pol[x]) * Dy(Pol[x]) - Kb * Dx(Pol[y]) * Dy(Pol[y]) + (Kb - Ks) * Dy(Pol[x]) * Dy(Pol[y]);
            sigma[y][y] =
                    -Ks * Dy(Pol[y]) * Dy(Pol[y]) - Kb * Dy(Pol[x]) * Dy(Pol[x]) + (Kb - Ks) * Dx(Pol[y]) * Dy(Pol[x]);
            Particles.ghost_get<Stress>(SKIP_LABELLING);


            r = Pol[x] * Pol[x] + Pol[y] * Pol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
            }

            h[y] = (Pol[x] * (Ks * Dyy(Pol[y]) + Kb * Dxx(Pol[y]) + (Ks - Kb) * Dxy(Pol[x])) -
                    Pol[y] * (Ks * Dxx(Pol[x]) + Kb * Dyy(Pol[x]) + (Ks - Kb) * Dxy(Pol[y])));

            Particles.ghost_get<MolField>(SKIP_LABELLING);

            FranckEnergyDensity = (Ks / 2.0) *
                                  ((Dx(Pol[x]) * Dx(Pol[x])) + (Dy(Pol[x]) * Dy(Pol[x])) +
                                   (Dx(Pol[y]) * Dx(Pol[y])) +
                                   (Dy(Pol[y]) * Dy(Pol[y]))) +
                                  ((Kb - Ks) / 2.0) * ((Dx(Pol[y]) - Dy(Pol[x])) * (Dx(Pol[y]) - Dy(Pol[x])));
            Particles.ghost_get<33>(SKIP_LABELLING);


            f1 = gama * nu * Pol[x] * Pol[x] * (Pol[x] * Pol[x] - Pol[y] * Pol[y]) / (r);
            f2 = 2.0 * gama * nu * Pol[x] * Pol[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y]) / (r);
            f3 = gama * nu * Pol[y] * Pol[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y]) / (r);
            f4 = 2.0 * gama * nu * Pol[x] * Pol[x] * Pol[x] * Pol[y] / (r);
            f5 = 4.0 * gama * nu * Pol[x] * Pol[x] * Pol[y] * Pol[y] / (r);
            f6 = 2.0 * gama * nu * Pol[x] * Pol[y] * Pol[y] * Pol[y] / (r);
            Particles.ghost_get<11, 12, 13, 14, 15, 16>(SKIP_LABELLING);
            Df1[x] = Dx(f1);
            Df2[x] = Dx(f2);
            Df3[x] = Dx(f3);
            Df4[x] = Dx(f4);
            Df5[x] = Dx(f5);
            Df6[x] = Dx(f6);

            Df1[y] = Dy(f1);
            Df2[y] = Dy(f2);
            Df3[y] = Dy(f3);
            Df4[y] = Dy(f4);
            Df5[y] = Dy(f5);
            Df6[y] = Dy(f6);
            Particles.ghost_get<21, 22, 23, 24, 25, 26>(SKIP_LABELLING);


            dV[x] = -0.5 * Dy(h[y]) + zeta * Dx(delmu * Pol[x] * Pol[x]) + zeta * Dy(delmu * Pol[x] * Pol[y]) -
                    zeta * Dx(0.5 * delmu * (Pol[x] * Pol[x] + Pol[y] * Pol[y])) -
                    0.5 * nu * Dx(-2.0 * h[y] * Pol[x] * Pol[y])
                    - 0.5 * nu * Dy(h[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y])) - Dx(sigma[x][x]) -
                    Dy(sigma[x][y]) -
                    g[x]
                    - 0.5 * nu * Dx(-gama * lambda * delmu * (Pol[x] * Pol[x] - Pol[y] * Pol[y]))
                    - 0.5 * Dy(-2.0 * gama * lambda * delmu * (Pol[x] * Pol[y]));


            dV[y] = -0.5 * Dx(-h[y]) + zeta * Dy(delmu * Pol[y] * Pol[y]) + zeta * Dx(delmu * Pol[x] * Pol[y]) -
                    zeta * Dy(0.5 * delmu * (Pol[x] * Pol[x] + Pol[y] * Pol[y])) -
                    0.5 * nu * Dy(2.0 * h[y] * Pol[x] * Pol[y])
                    - 0.5 * nu * Dx(h[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y])) - Dx(sigma[y][x]) -
                    Dy(sigma[y][y]) -
                    g[y]
                    - 0.5 * nu * Dy(gama * lambda * delmu * (Pol[x] * Pol[x] - Pol[y] * Pol[y]))
                    - 0.5 * Dx(-2.0 * gama * lambda * delmu * (Pol[x] * Pol[y]));
            Particles.ghost_get<9>(SKIP_LABELLING);


            //Particles.write("PolarI");
            //Velocity Solution n iterations


            auto Stokes1 = eta * (Dxx(V[x]) + Dyy(V[x]))
                           + 0.5 * nu * (Df1[x] * Dx(V[x]) + f1 * Dxx(V[x]))
                           + 0.5 * nu * (Df2[x] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f2 * 0.5 * (Dxx(V[y]) + Dyx(V[x])))
                           + 0.5 * nu * (Df3[x] * Dy(V[y]) + f3 * Dyx(V[y]))
                           + 0.5 * nu * (Df4[y] * Dx(V[x]) + f4 * Dxy(V[x]))
                           + 0.5 * nu * (Df5[y] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f5 * 0.5 * (Dxy(V[y]) + Dyy(V[x])))
                           + 0.5 * nu * (Df6[y] * Dy(V[y]) + f6 * Dyy(V[y]));
            auto Stokes2 = eta * (Dxx(V[y]) + Dyy(V[y]))
                           - 0.5 * nu * (Df1[y] * Dx(V[x]) + f1 * Dxy(V[x]))
                           - 0.5 * nu * (Df2[y] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f2 * 0.5 * (Dxy(V[y]) + Dyy(V[x])))
                           - 0.5 * nu * (Df3[y] * Dy(V[y]) + f3 * Dyy(V[y]))
                           + 0.5 * nu * (Df4[x] * Dx(V[x]) + f4 * Dxx(V[x]))
                           + 0.5 * nu * (Df5[x] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f5 * 0.5 * (Dxx(V[y]) + Dyx(V[x])))
                           + 0.5 * nu * (Df6[x] * Dy(V[y]) + f6 * Dyx(V[y]));
            tt.stop();
            std::cout << "Init of Velocity took " << tt.getwct() << " seconds." << std::endl;
            tt.start();
            V_err = 1;
            n = 0;
            errctr = 0;
            if (Vreset == 1) {
                P_bulk = 0;
                P = 0;
                Vreset = 0;
            }
            P = 0;
            P_bulk = 0;

            //////////////////////// DEBUG test //////////////////////////

//            DCPSE_scheme<equations2d2, decltype(Particles)> Solver(Particles);
            /*PLAN FOR HACKATHON


            P=Dx(V);
            Pbulk=Dx(V);
            Pbulk=Bulk_Dx(V);



             */

//            DCPSE_scheme<equations2d1, decltype(Particles)> SolverH(Particles);

            //////////////////////////////////////////////////////////////

            while (V_err >= V_err_eps && n <= nmax) {
                RHS[x] = dV[x];
                RHS[y] = dV[y];
                Particles_subset.ghost_get<0>(SKIP_LABELLING);
                Grad_bulk[x] = Bulk_Dx(P_bulk);
                Grad_bulk[y] = Bulk_Dy(P_bulk);
                for (int i = 0; i < bulk.size(); i++) {
                    Particles.template getProp<10>(bulk.template get<0>(i))[x] += Particles_subset.getProp<2>(i)[x];
                    Particles.template getProp<10>(bulk.template get<0>(i))[y] += Particles_subset.getProp<2>(i)[y];
                }
                Particles.ghost_get<10>(SKIP_LABELLING);
                DCPSE_scheme<equations2d2, decltype(Particles)> Solver(Particles);
//                Solver.reset(Particles);
                Solver.impose(Stokes1, bulk, RHS[0], vx);
                Solver.impose(Stokes2, bulk, RHS[1], vy);
                Solver.impose(V[x], up_p, 0, vx);
                Solver.impose(V[y], up_p, 0, vy);
                Solver.impose(V[x], dw_p, 0, vx);
                Solver.impose(V[y], dw_p, 0, vy);
                Solver.impose(V[x], l_p, 0, vx);
                Solver.impose(V[y], l_p, 0, vy);
                Solver.impose(V[x], r_p, 0, vx);
                Solver.impose(V[y], r_p, 0, vy);
                Solver.solve_with_solver(solverPetsc, V[x], V[y]);
                //Solver.solve(V[x], V[y]);
                Particles.ghost_get<Velocity>(SKIP_LABELLING);
                div = -(Dx(V[x]) + Dy(V[y]));
                auto Helmholtz = Dxx(H) + Dyy(H);
                DCPSE_scheme<equations2d1, decltype(Particles)> SolverH(Particles);
//                SolverH.reset(Particles);
                SolverH.impose(Helmholtz, bulk, prop_id<19>());
                SolverH.impose(H, up_p1, 0);
                SolverH.impose(H, dw_p1, 0);
                SolverH.impose(H, l_p1, 0);
                SolverH.impose(H, r_p1, 0);
                SolverH.impose(-Dx(H) + Dy(H), corner_ul, 0);
                SolverH.impose(Dx(H) + Dy(H), corner_ur, 0);
                SolverH.impose(-Dx(H) - Dy(H), corner_dl, 0);
                SolverH.impose(Dx(H) - Dy(H), corner_dr, 0);
                SolverH.solve_with_solver(solverPetsc2, H);
                //SolverH.solve(H);
                P = P + div;
                for (int i = 0; i < bulk.size(); i++) {
                    Particles_subset.getProp<0>(i) = Particles.template getProp<4>(bulk.template get<0>(i));
                }
                for (int j = 0; j < up_p.size(); j++) {
                    auto p = up_p.get<0>(j);
                    Particles.getProp<4>(p) = 0;

                }
                for (int j = 0; j < dw_p.size(); j++) {
                    auto p = dw_p.get<0>(j);
                    Particles.getProp<4>(p) = 0;
                }
                for (int j = 0; j < l_p.size(); j++) {
                    auto p = l_p.get<0>(j);
                    Particles.getProp<4>(p) = 0;
                }
                for (int j = 0; j < r_p.size(); j++) {
                    auto p = r_p.get<0>(j);
                    Particles.getProp<4>(p) = 0;
                }
                sum = 0;
                sum1 = 0;
                for (int j = 0; j < bulk.size(); j++) {
                    auto p = bulk.get<0>(j);
                    sum += (Particles.getProp<18>(p)[0] - Particles.getProp<1>(p)[0]) *
                           (Particles.getProp<18>(p)[0] - Particles.getProp<1>(p)[0]) +
                           (Particles.getProp<18>(p)[1] - Particles.getProp<1>(p)[1]) *
                           (Particles.getProp<18>(p)[1] - Particles.getProp<1>(p)[1]);
                    sum1 += Particles.getProp<1>(p)[0] * Particles.getProp<1>(p)[0] +
                            Particles.getProp<1>(p)[1] * Particles.getProp<1>(p)[1];
                }
                sum = sqrt(sum);
                sum1 = sqrt(sum1);

                v_cl.sum(sum);
                v_cl.sum(sum1);
                v_cl.execute();
                V_t = V;
                Particles.ghost_get<1,4,18>(SKIP_LABELLING);
                V_err_old = V_err;
                V_err = sum / sum1;
                if (V_err > V_err_old || abs(V_err_old - V_err) < 1e-8) {
                    errctr++;
                    //alpha_P -= 0.1;
                } else {
                    errctr = 0;
                }
                if (n > 3) {
                    if (errctr > 3) {
                        std::cout << "CONVERGENCE LOOP BROKEN DUE TO INCREASE/VERY SLOW DECREASE IN ERROR" << std::endl;
                        Vreset = 1;
                        break;
                    } else {
                        Vreset = 0;
                    }
                }
                n++;
                //Particles.write_frame("V_debug", n);
                if (v_cl.rank() == 0) {
                    std::cout << "Rel l2 cgs err in V = " << V_err << " at " << n << std::endl;
                }
            }
            tt.stop();
            std::cout << "Rel l2 cgs err in V = " << V_err << " and took " << tt.getwct() << " seconds with " << n
                      << " iterations."
                      << std::endl;
            u[x][x] = Dx(V[x]);
            u[x][y] = 0.5 * (Dx(V[y]) + Dy(V[x]));
            u[y][x] = 0.5 * (Dy(V[x]) + Dx(V[y]));
            u[y][y] = Dy(V[y]);

            W[x][x] = 0;
            W[x][y] = 0.5 * (Dy(V[x]) - Dx(V[y]));
            W[y][x] = 0.5 * (Dx(V[y]) - Dy(V[x]));
            W[y][y] = 0;

            H_p_b = Pol[x] * Pol[x] + Pol[y] * Pol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<32>(p) = (Particles.getProp<32>(p) == 0) ? 1 : Particles.getProp<32>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<32>(p) = (Particles.getProp<32>(p) == 0) ? 1 : Particles.getProp<32>(p);
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<32>(p) = (Particles.getProp<32>(p) == 0) ? 1 : Particles.getProp<32>(p);
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<32>(p) = (Particles.getProp<32>(p) == 0) ? 1 : Particles.getProp<32>(p);
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<32>(p) = (Particles.getProp<32>(p) == 0) ? 1 : Particles.getProp<32>(p);
            }

            h[x] = -gama * (lambda * delmu - nu * (u[x][x] * Pol[x] * Pol[x] + u[y][y] * Pol[y] * Pol[y] +
                                                   2 * u[x][y] * Pol[x] * Pol[y]) / (H_p_b));


            //Particles.ghost_get<MolField, Strain_rate, Vorticity>(SKIP_LABELLING);
            //Particles.write_frame("Polar_withGhost_3e-3", ctr);
            Particles.deleteGhost();
            Particles.write_frame("Polar_3e-3", ctr);
            Particles.ghost_get<0>();

            ctr++;

            H_p_b = sqrt(H_p_b);

            k1[x] = ((h[x] * Pol[x] - h[y] * Pol[y]) / gama + lambda * delmu * Pol[x] -
                     nu * (u[x][x] * Pol[x] + u[x][y] * Pol[y]) + W[x][x] * Pol[x] +
                     W[x][y] * Pol[y]);// - V[x] * Dx(Pol[x]) - V[y] * Dy(Pol[x]));
            k1[y] = ((h[x] * Pol[y] + h[y] * Pol[x]) / gama + lambda * delmu * Pol[y] -
                     nu * (u[y][x] * Pol[x] + u[y][y] * Pol[y]) + W[y][x] * Pol[x] +
                     W[y][y] * Pol[y]);// - V[x] * Dx(Pol[y]) - V[y] * Dy(Pol[y]));

            H_t = H_p_b;//+0.5*dt*(k1[x]*k1[x]+k1[y]*k1[y]);
            dPol = Pol + (0.5 * dt) * k1;
            dPol = dPol / H_t;
            r = dPol[x] * dPol[x] + dPol[y] * dPol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            Particles.ghost_get<8>(SKIP_LABELLING);


            h[y] = (dPol[x] * (Ks * Dyy(dPol[y]) + Kb * Dxx(dPol[y]) + (Ks - Kb) * Dxy(dPol[x])) -
                    dPol[y] * (Ks * Dxx(dPol[x]) + Kb * Dyy(dPol[x]) + (Ks - Kb) * Dxy(dPol[y])));

            h[x] = -gama * (lambda * delmu - nu * ((u[x][x] * dPol[x] * dPol[x] + u[y][y] * dPol[y] * dPol[y] +
                                                    2 * u[x][y] * dPol[x] * dPol[y]) / (r)));

            k2[x] = ((h[x] * (dPol[x]) - h[y] * (dPol[y])) / gama +
                     lambda * delmu * (dPol[x]) -
                     nu * (u[x][x] * (dPol[x]) + u[x][y] * (dPol[y])) +
                     W[x][x] * (dPol[x]) + W[x][y] * (dPol[y])); //-V[x] * Dx((dPol[x])) - V[y] * Dy((dPol[x])));
            k2[y] = ((h[x] * (dPol[y]) + h[y] * (dPol[x])) / gama +
                     lambda * delmu * (dPol[y]) -
                     nu * (u[y][x] * (dPol[x]) + u[y][y] * (dPol[y])) +
                     W[y][x] * (dPol[x]) + W[y][y] * (dPol[y])); //-V[x] * Dx((dPol[y])) - V[y] * Dy((dPol[y])));

            H_t = H_p_b;//+0.5*dt*(k2[x]*k2[x]+k2[y]*k2[y]);
            dPol = Pol + (0.5 * dt) * k2;
            dPol = dPol / H_t;
            r = dPol[x] * dPol[x] + dPol[y] * dPol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            Particles.ghost_get<8>(SKIP_LABELLING);

            h[y] = (dPol[x] * (Ks * Dyy(dPol[y]) + Kb * Dxx(dPol[y]) + (Ks - Kb) * Dxy(dPol[x])) -
                    dPol[y] * (Ks * Dxx(dPol[x]) + Kb * Dyy(dPol[x]) + (Ks - Kb) * Dxy(dPol[y])));

            h[x] = -gama * (lambda * delmu - nu * ((u[x][x] * dPol[x] * dPol[x] + u[y][y] * dPol[y] * dPol[y] +
                                                    2 * u[x][y] * dPol[x] * dPol[y]) / (r)));

            k3[x] = ((h[x] * (dPol[x]) - h[y] * (dPol[y])) / gama +
                     lambda * delmu * (dPol[x]) -
                     nu * (u[x][x] * (dPol[x]) + u[x][y] * (dPol[y])) +
                     W[x][x] * (dPol[x]) + W[x][y] * (dPol[y]));
            // -V[x] * Dx((dPol[x])) - V[y] * Dy((dPol[x])));
            k3[y] = ((h[x] * (dPol[y]) + h[y] * (dPol[x])) / gama +
                     lambda * delmu * (dPol[y]) -
                     nu * (u[y][x] * (dPol[x]) + u[y][y] * (dPol[y])) +
                     W[y][x] * (dPol[x]) + W[y][y] * (dPol[y]));
            // -V[x] * Dx((dPol[y])) - V[y] * Dy((dPol[y])));
            H_t = H_p_b;//+dt*(k3[x]*k3[x]+k3[y]*k3[y]);
            dPol = Pol + (dt * k3);
            dPol = dPol / H_t;
            Particles.ghost_get<8>(SKIP_LABELLING);
            r = dPol[x] * dPol[x] + dPol[y] * dPol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));

            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            Particles.ghost_get<8>(SKIP_LABELLING);

            h[y] = (dPol[x] * (Ks * Dyy(dPol[y]) + Kb * Dxx(dPol[y]) + (Ks - Kb) * Dxy(dPol[x])) -
                    dPol[y] * (Ks * Dxx(dPol[x]) + Kb * Dyy(dPol[x]) + (Ks - Kb) * Dxy(dPol[y])));

            h[x] = -gama * (lambda * delmu - nu * ((u[x][x] * dPol[x] * dPol[x] + u[y][y] * dPol[y] * dPol[y] +
                                                    2 * u[x][y] * dPol[x] * dPol[y]) / (r)));

            k4[x] = ((h[x] * (dPol[x]) - h[y] * (dPol[y])) / gama +
                     lambda * delmu * (dPol[x]) -
                     nu * (u[x][x] * (dPol[x]) + u[x][y] * (dPol[y])) +
                     W[x][x] * (dPol[x]) +
                     W[x][y] * (dPol[y]));//   -V[x]*Dx( (dt * k3[x]+Pol[x])) -V[y]*Dy( (dt * k3[x]+Pol[x])));
            k4[y] = ((h[x] * (dPol[y]) + h[y] * (dPol[x])) / gama +
                     lambda * delmu * (dPol[y]) -
                     nu * (u[y][x] * (dPol[x]) + u[y][y] * (dPol[y])) +
                     W[y][x] * (dPol[x]) +
                     W[y][y] * (dPol[y]));//  -V[x]*Dx( (dt * k3[y]+Pol*[y])) -V[y]*Dy( (dt * k3[y]+Pol[y])));

            Pol = Pol + (dt / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4);
            Pol = Pol / H_p_b;

            H_p_b = sqrt(Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            Pol = Pol / H_p_b;
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }

            k1 = V;
            k2 = 0.5 * dt * k1 + V;
            k3 = 0.5 * dt * k2 + V;
            k4 = dt * k3 + V;
            //Pos = Pos + dt * V;
            Pos = Pos + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);

            Particles.map();

            Particles.ghost_get<0, ExtForce, 27>();
            indexUpdate(Particles, Particles_subset, up_p, dw_p, l_p, r_p, up_p1, dw_p1, l_p1, r_p1, corner_ul,
                        corner_ur, corner_dl, corner_dr, bulk, up, down, left, right);
            Particles_subset.map();
            Particles_subset.ghost_get<0>();

            //Particles_subset.write("debug");

            tt.start();
            Dx.update(Particles);
            Dy.update(Particles);
            Dxy.update(Particles);
            auto Dyx = Dxy;
            Dxx.update(Particles);
            Dyy.update(Particles);

            Bulk_Dx.update(Particles_subset);
            Bulk_Dy.update(Particles_subset);

            tt.stop();
            std::cout << "Updation of operators took " << tt.getwct() << " seconds." << std::endl;

            std::cout << "Time step " << ctr - 1 << " : " << tim << " over." << std::endl;
            tim += dt;
            std::cout << "----------------------------------------------------------" << std::endl;
        }

        Dx.deallocate(Particles);
        Dy.deallocate(Particles);
        Dxy.deallocate(Particles);
        Dxx.deallocate(Particles);
        Dyy.deallocate(Particles);
        Bulk_Dx.deallocate(Particles_subset);
        Bulk_Dy.deallocate(Particles_subset);
        Particles.deleteGhost();
        tt2.stop();
        std::cout << "The simulation took " << tt2.getwct() << "Seconds.";
    }


    BOOST_AUTO_TEST_CASE(Active2DExp) {
        timer tt2;
        tt2.start();
        double boxsize = 10;
        const size_t sz[2] = {41, 41};
        Box<2, double> box({0, 0}, {boxsize, boxsize});
        double Lx = box.getHigh(0);
        double Ly = box.getHigh(1);
        size_t bc[2] = {NON_PERIODIC, NON_PERIODIC};
        double spacing = box.getHigh(0) / (sz[0] - 1);
        double rCut = 3.9 * spacing;
        double rCut2 = 3.9 * spacing;
        double rCut3 = 5 * spacing;

        int ord = 2;
        int ord2 = 2;
        double sampling_factor = 4.0;
        double sampling_factor2 = 2.4;
        double sampling_factor3 = 1.6;

        double alpha_V = 1.0;
        double alpha_P = 1.0;
        Ghost<2, double> ghost(rCut3);

        auto &v_cl = create_vcluster();

/*                                          pol                             V         vort                 Ext    Press     strain       stress                      Mfield,   dPol                      dV         RHS                  f1     f2     f3    f4     f5     f6       H               V_t      div    H_t                                                                                      delmu */
        vector_dist<2, double, aggregate<VectorS<2, double>, VectorS<2, double>, double[2][2], VectorS<2, double>, double, double[2][2], double[2][2], VectorS<2, double>, VectorS<2, double>, VectorS<2, double>, VectorS<2, double>, double, double, double, double, double, double, double, VectorS<2, double>, double, double, double[2], double[2], double[2], double[2], double[2], double[2], double, VectorS<2, double>, VectorS<2, double>, VectorS<2, double>, VectorS<2, double>, double, double, double>> Particles(
                0, box, bc, ghost);
        vector_dist<2, double, aggregate<double, double, VectorS<2, double>>> Particles_subset(
                Particles.getDecomposition(), 0);
        double x0, y0, x1, y1;
        x0 = box.getLow(0);
        y0 = box.getLow(1);
        x1 = box.getHigh(0);
        y1 = box.getHigh(1);

        auto it = Particles.getGridIterator(sz);
        while (it.isNext()) {
            Particles.add();
            auto key = it.get();
            double x = key.get(0) * it.getSpacing(0);
            Particles.getLastPos()[0] = x;
            double y = key.get(1) * it.getSpacing(1);
            Particles.getLastPos()[1] = y;
            ++it;
        }

        BOOST_TEST_MESSAGE("Sync domain across processors...");

        Particles.map();
        Particles.ghost_get<0>();

        Particles.write("Par");

        openfpm::vector<aggregate<int>> bulk;
        openfpm::vector<aggregate<int>> bulk_withoutmid;
        openfpm::vector<aggregate<int>> mid_ref;
        openfpm::vector<aggregate<int>> up_p;
        openfpm::vector<aggregate<int>> dw_p;
        openfpm::vector<aggregate<int>> l_p;
        openfpm::vector<aggregate<int>> r_p;
        openfpm::vector<aggregate<int>> up_p1;
        openfpm::vector<aggregate<int>> dw_p1;
        openfpm::vector<aggregate<int>> l_p1;
        openfpm::vector<aggregate<int>> r_p1;
        openfpm::vector<aggregate<int>> corner_ul;
        openfpm::vector<aggregate<int>> corner_ur;
        openfpm::vector<aggregate<int>> corner_dl;
        openfpm::vector<aggregate<int>> corner_dr;


        constexpr int x = 0;
        constexpr int y = 1;

        constexpr int Polarization = 0;
        constexpr int Velocity = 1;
        constexpr int Vorticity = 2;
        constexpr int ExtForce = 3;
        constexpr int Pressure = 4;
        constexpr int Strain_rate = 5;
        constexpr int Stress = 6;
        constexpr int MolField = 7;
        auto Pos = getV<PROP_POS>(Particles);

        auto Pol = getV<Polarization>(Particles);
        auto V = getV<Velocity>(Particles);
        auto W = getV<Vorticity>(Particles);
        auto g = getV<ExtForce>(Particles);
        auto P = getV<Pressure>(Particles);
        auto P_bulk = getV<0>(Particles_subset); //Pressure only on inside
        auto u = getV<Strain_rate>(Particles);
        auto sigma = getV<Stress>(Particles);
        auto h = getV<MolField>(Particles);
        auto dPol = getV<8>(Particles);
        auto dV = getV<9>(Particles);
        auto RHS = getV<10>(Particles);
        auto f1 = getV<11>(Particles);
        auto f2 = getV<12>(Particles);
        auto f3 = getV<13>(Particles);
        auto f4 = getV<14>(Particles);
        auto f5 = getV<15>(Particles);
        auto f6 = getV<16>(Particles);
        auto H = getV<17>(Particles);
        auto H_bulk = getV<1>(Particles_subset); //Pressure only on inside
        auto Grad_bulk = getV<2>(Particles_subset);

        auto V_t = getV<18>(Particles);
        auto div = getV<19>(Particles);
        auto H_t = getV<20>(Particles);
        auto Df1 = getV<21>(Particles);
        auto Df2 = getV<22>(Particles);
        auto Df3 = getV<23>(Particles);
        auto Df4 = getV<24>(Particles);
        auto Df5 = getV<25>(Particles);
        auto Df6 = getV<26>(Particles);
        auto delmu = getV<27>(Particles);
        auto k1 = getV<28>(Particles);
        auto k2 = getV<29>(Particles);
        auto k3 = getV<30>(Particles);
        auto k4 = getV<31>(Particles);
        auto H_p_b = getV<32>(Particles);
        auto FranckEnergyDensity = getV<33>(Particles);
        auto r = getV<34>(Particles);


        double eta = 1.0;
        double nu = -0.5;
        double gama = 0.1;
        double zeta = 0.07;
        double Ks = 1.0;
        double Kb = 1.0;
        double lambda = 0.1;
        //double delmu = -1.0;
        g = 0;
        delmu = -1.0;
        P = 0;
        P_bulk = 0;
        V = 0;
        // Here fill up the boxes for particle boundary detection.
        Particles.ghost_get<ExtForce, 27>(SKIP_LABELLING);


        Box<2, double> up({x0 - spacing / 2.0, y1 - spacing / 2.0},
                          {x1 + spacing / 2.0, y1 + spacing / 2.0});

        Box<2, double> down({x0 - spacing / 2.0, y0 - spacing / 2.0},
                            {x1 + spacing / 2.0, y0 + spacing / 2.0});

        Box<2, double> left({x0 - spacing / 2.0, y0 - spacing / 2.0},
                            {x0 + spacing / 2.0, y1 + spacing / 2.0});

        Box<2, double> right({x1 - spacing / 2.0, y0 - spacing / 2.0},
                             {x1 + spacing / 2.0, y1 + spacing / 2.0});
        Box<2, double> mid({box.getHigh(0) / 2.0 - 0.75*spacing, box.getHigh(1) / 2.0 - 0.75*spacing },
                           {box.getHigh(0) / 2.0+0.75*spacing, box.getHigh(1) / 2.0 + 0.75*spacing});


        openfpm::vector<Box<2, double>> boxes;
        boxes.add(up);
        boxes.add(down);
        boxes.add(left);
        boxes.add(right);
        boxes.add(mid);


        VTKWriter<openfpm::vector<Box<2, double>>, VECTOR_BOX> vtk_box;
        vtk_box.add(boxes);
        vtk_box.write("vtk_box.vtk");

        auto it2 = Particles.getDomainIterator();

        while (it2.isNext()) {
            auto p = it2.get();
            Point<2, double> xp = Particles.getPos(p);
            Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * xp[x] - Lx) / Lx) - sin((2 * xp[y] - Ly) / Ly)));
            Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * xp[x] - Lx) / Lx) - sin((2 * xp[y] - Ly) / Ly)));
            if (up.isInside(xp) == true) {
                if (left.isInside(xp) == true) {
                    corner_ul.add();
                    corner_ul.last().get<0>() = p.getKey();
                } else if (right.isInside(xp) == true) {
                    corner_ur.add();
                    corner_ur.last().get<0>() = p.getKey();
                } else {
                    up_p1.add();
                    up_p1.last().get<0>() = p.getKey();
                }
                up_p.add();
                up_p.last().get<0>() = p.getKey();

            } else if (down.isInside(xp) == true) {
                if (left.isInside(xp) == true) {
                    corner_dl.add();
                    corner_dl.last().get<0>() = p.getKey();
                } else if (right.isInside(xp) == true) {
                    corner_dr.add();
                    corner_dr.last().get<0>() = p.getKey();
                } else {
                    dw_p1.add();
                    dw_p1.last().get<0>() = p.getKey();
                }
                dw_p.add();
                dw_p.last().get<0>() = p.getKey();
            } else if (left.isInside(xp) == true) {
                if (up.isInside(xp) == false && down.isInside(xp) == false) {
                    l_p1.add();
                    l_p1.last().get<0>() = p.getKey();
                }
                l_p.add();
                l_p.last().get<0>() = p.getKey();
            } else if (right.isInside(xp) == true) {
                if (up.isInside(xp) == false && down.isInside(xp) == false) {
                    r_p1.add();
                    r_p1.last().get<0>() = p.getKey();
                }
                r_p.add();
                r_p.last().get<0>() = p.getKey();
            } else {
                if (mid.isInside(xp) == true) {
                    mid_ref.add();
                    mid_ref.last().get<0>() = p.getKey();
                } else{
                    bulk_withoutmid.add();
                    bulk_withoutmid.last().get<0>() = p.getKey();
                }
                bulk.add();
                bulk.last().get<0>() = p.getKey();
            }
            ++it2;
        }

        double sigma2 = spacing * spacing / (4);
        std::mt19937 rng{7};
        std::normal_distribution<> gaussian{0, sigma2};


        for(int j=0;j<bulk.size();j++) {
            auto p = bulk.get<0>(j);
            Particles.getPos(p)[0]=Particles.getPos(p)[0]+gaussian(rng);
            Particles.getPos(p)[1]=Particles.getPos(p)[1]+gaussian(rng);
        }


        for (int i = 0; i < bulk.size(); i++) {
            Particles_subset.add();
            Particles_subset.getLastPos()[0] = Particles.getPos(bulk.template get<0>(i))[0];
            Particles_subset.getLastPos()[1] = Particles.getPos(bulk.template get<0>(i))[1];
        }


        Particles_subset.map();
        Particles_subset.ghost_get<0>();

        //Particles_subset.write("Pars");
        Derivative_x Dx(Particles, ord, rCut, sampling_factor, support_options::RADIUS), Bulk_Dx(Particles_subset, ord,
                                                                                                 rCut,
                                                                                                 sampling_factor,
                                                                                                 support_options::RADIUS);
        Derivative_y Dy(Particles, ord, rCut, sampling_factor, support_options::RADIUS), Bulk_Dy(Particles_subset, ord,
                                                                                                 rCut,
                                                                                                 sampling_factor,
                                                                                                 support_options::RADIUS);
        Derivative_xy Dxy(Particles, ord, rCut2, sampling_factor2, support_options::RADIUS);
        auto Dyx = Dxy;
        Derivative_xx Dxx(Particles, ord, rCut2, sampling_factor2,
                          support_options::RADIUS);//, Dxx2(Particles, ord2, rCut2, sampling_factor2, support_options::RADIUS);
        Derivative_yy Dyy(Particles, ord, rCut2, sampling_factor2,
                          support_options::RADIUS);//, Dyy2(Particles, ord2, rCut2, sampling_factor2, support_options::RADIUS);


        Derivative_xxx Dxxx(Particles, ord, rCut3, sampling_factor3,
                            support_options::RADIUS);
        Derivative_xxy Dxxy(Particles, ord, rCut3, sampling_factor3,
                            support_options::RADIUS);

        Derivative_yyx Dyyx(Particles, ord, rCut3, sampling_factor3,
                            support_options::RADIUS);

        Derivative_yyy Dyyy(Particles, ord, rCut3, sampling_factor3,
                            support_options::RADIUS);

/*        Derivative_x Dx(Particles, ord, rCut, sampling_factor), Bulk_Dx(Particles_subset, ord, rCut, sampling_factor);
        Derivative_y Dy(Particles, ord, rCut, sampling_factor), Bulk_Dy(Particles_subset, ord, rCut, sampling_factor);
        Derivative_xy Dxy(Particles, ord2, rCut, sampling_factor);
        auto Dyx = Dxy;
        Derivative_xx Dxx(Particles, ord2, rCut2, sampling_factor2),Bulk_Dxx(Particles_subset, ord2, rCut2, sampling_factor2);
        Derivative_yy Dyy(Particles, ord2, rCut2, sampling_factor2),Bulk_Dyy(Particles_subset, ord2, rCut2, sampling_factor2);*/

        V.setVarId(0);
        P.setVarId(2);
        H.setVarId(3);
        eq_id vx, vy, ic,helm;
        timer tt;
        timer tt3;
        vx.setId(0);
        vy.setId(1);
        ic.setId(2);
        helm.setId(3);
        double V_err_eps = 5 * 1e-2;
        double V_err = 1, V_err_old;
        int n = 0;
        int nmax = 300;
        int ctr = 0, errctr, Vreset = 0;
        double dt = 3 * 1e-3;

        double tim = 0;
        double tf = 1.25;
        div = 0;
        double sum, sum1, sum_k;
        while (tim <= tf) {
            tt.start();
            /*petsc_solver<double> solverPetsc;
            solverPetsc.setSolver(KSPGMRES);
            //solverPetsc.setRestart(250);
            solverPetsc.setPreconditioner(PCJACOBI);*/

            Particles.ghost_get<Polarization>(SKIP_LABELLING);
            sigma[x][x] =
                    -Ks * Dx(Pol[x]) * Dx(Pol[x]) - Kb * Dx(Pol[y]) * Dx(Pol[y]) + (Kb - Ks) * Dy(Pol[x]) * Dx(Pol[y]);
            sigma[x][y] =
                    -Ks * Dy(Pol[y]) * Dx(Pol[y]) - Kb * Dy(Pol[x]) * Dx(Pol[x]) + (Kb - Ks) * Dx(Pol[y]) * Dx(Pol[x]);
            sigma[y][x] =
                    -Ks * Dx(Pol[x]) * Dy(Pol[x]) - Kb * Dx(Pol[y]) * Dy(Pol[y]) + (Kb - Ks) * Dy(Pol[x]) * Dy(Pol[y]);
            sigma[y][y] =
                    -Ks * Dy(Pol[y]) * Dy(Pol[y]) - Kb * Dy(Pol[x]) * Dy(Pol[x]) + (Kb - Ks) * Dx(Pol[y]) * Dy(Pol[x]);
            Particles.ghost_get<Stress>(SKIP_LABELLING);


            r = Pol[x] * Pol[x] + Pol[y] * Pol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
            }

            h[y] = (Pol[x] * (Ks * Dyy(Pol[y]) + Kb * Dxx(Pol[y]) + (Ks - Kb) * Dxy(Pol[x])) -
                    Pol[y] * (Ks * Dxx(Pol[x]) + Kb * Dyy(Pol[x]) + (Ks - Kb) * Dxy(Pol[y])));

            Particles.ghost_get<MolField>(SKIP_LABELLING);

            FranckEnergyDensity = (Ks / 2.0) *
                                  ((Dx(Pol[x]) * Dx(Pol[x])) + (Dy(Pol[x]) * Dy(Pol[x])) +
                                   (Dx(Pol[y]) * Dx(Pol[y])) +
                                   (Dy(Pol[y]) * Dy(Pol[y]))) +
                                  ((Kb - Ks) / 2.0) * ((Dx(Pol[y]) - Dy(Pol[x])) * (Dx(Pol[y]) - Dy(Pol[x])));
            Particles.ghost_get<33>(SKIP_LABELLING);


            f1 = gama * nu * Pol[x] * Pol[x] * (Pol[x] * Pol[x] - Pol[y] * Pol[y]) / (r);
            f2 = 2.0 * gama * nu * Pol[x] * Pol[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y]) / (r);
            f3 = gama * nu * Pol[y] * Pol[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y]) / (r);
            f4 = 2.0 * gama * nu * Pol[x] * Pol[x] * Pol[x] * Pol[y] / (r);
            f5 = 4.0 * gama * nu * Pol[x] * Pol[x] * Pol[y] * Pol[y] / (r);
            f6 = 2.0 * gama * nu * Pol[x] * Pol[y] * Pol[y] * Pol[y] / (r);
            Particles.ghost_get<11, 12, 13, 14, 15, 16>(SKIP_LABELLING);
            Df1[x] = Dx(f1);
            Df2[x] = Dx(f2);
            Df3[x] = Dx(f3);
            Df4[x] = Dx(f4);
            Df5[x] = Dx(f5);
            Df6[x] = Dx(f6);

            Df1[y] = Dy(f1);
            Df2[y] = Dy(f2);
            Df3[y] = Dy(f3);
            Df4[y] = Dy(f4);
            Df5[y] = Dy(f5);
            Df6[y] = Dy(f6);
            Particles.ghost_get<21, 22, 23, 24, 25, 26>(SKIP_LABELLING);


            dV[x] = -0.5 * Dy(h[y]) + zeta * Dx(delmu * Pol[x] * Pol[x]) + zeta * Dy(delmu * Pol[x] * Pol[y]) -
                    zeta * Dx(0.5 * delmu * (Pol[x] * Pol[x] + Pol[y] * Pol[y])) -
                    0.5 * nu * Dx(-2.0 * h[y] * Pol[x] * Pol[y])
                    - 0.5 * nu * Dy(h[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y])) - Dx(sigma[x][x]) -
                    Dy(sigma[x][y]) -
                    g[x]
                    - 0.5 * nu * Dx(-gama * lambda * delmu * (Pol[x] * Pol[x] - Pol[y] * Pol[y]))
                    - 0.5 * Dy(-2.0 * gama * lambda * delmu * (Pol[x] * Pol[y]));


            dV[y] = -0.5 * Dx(-h[y]) + zeta * Dy(delmu * Pol[y] * Pol[y]) + zeta * Dx(delmu * Pol[x] * Pol[y]) -
                    zeta * Dy(0.5 * delmu * (Pol[x] * Pol[x] + Pol[y] * Pol[y])) -
                    0.5 * nu * Dy(2.0 * h[y] * Pol[x] * Pol[y])
                    - 0.5 * nu * Dx(h[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y])) - Dx(sigma[y][x]) -
                    Dy(sigma[y][y]) -
                    g[y]
                    - 0.5 * nu * Dy(gama * lambda * delmu * (Pol[x] * Pol[x] - Pol[y] * Pol[y]))
                    - 0.5 * Dx(-2.0 * gama * lambda * delmu * (Pol[x] * Pol[y]));
            Particles.ghost_get<9>(SKIP_LABELLING);


            //Particles.write("PolarI");
            //Velocity Solution n iterations


/*            auto Stokes1 = -(Dxxx(H) + Dyyx(H)) + eta * (Dxx(V[x]) + Dyy(V[x]))
                           + 0.5 * nu * (Df1[x] * Dx(V[x]) + f1 * Dxx(V[x]))
                           + 0.5 * nu * (Df2[x] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f2 * 0.5 * (Dxx(V[y]) + Dyx(V[x])))
                           + 0.5 * nu * (Df3[x] * Dy(V[y]) + f3 * Dyx(V[y]))
                           + 0.5 * nu * (Df4[y] * Dx(V[x]) + f4 * Dxy(V[x]))
                           + 0.5 * nu * (Df5[y] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f5 * 0.5 * (Dxy(V[y]) + Dyy(V[x])))
                           + 0.5 * nu * (Df6[y] * Dy(V[y]) + f6 * Dyy(V[y]));
            auto Stokes2 = -(Dxxy(H) + Dyyy(H)) + eta * (Dxx(V[y]) + Dyy(V[y]))
                           - 0.5 * nu * (Df1[y] * Dx(V[x]) + f1 * Dxy(V[x]))
                           - 0.5 * nu * (Df2[y] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f2 * 0.5 * (Dxy(V[y]) + Dyy(V[x])))
                           - 0.5 * nu * (Df3[y] * Dy(V[y]) + f3 * Dyy(V[y]))
                           + 0.5 * nu * (Df4[x] * Dx(V[x]) + f4 * Dxx(V[x]))
                           + 0.5 * nu * (Df5[x] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f5 * 0.5 * (Dxx(V[y]) + Dyx(V[x])))
                           + 0.5 * nu * (Df6[x] * Dy(V[y]) + f6 * Dyx(V[y]));

            auto Helmholtz = Dxx(H) + Dyy(H)+Dx(Stokes1) + Dy(Stokes2);*/

            auto Stokes1 = -(Bulk_Dx(P)) + eta * (Dxx(V[x]) + Dyy(V[x]))
                           + 0.5 * nu * (Df1[x] * Dx(V[x]) + f1 * Dxx(V[x]))
                           + 0.5 * nu * (Df2[x] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f2 * 0.5 * (Dxx(V[y]) + Dyx(V[x])))
                           + 0.5 * nu * (Df3[x] * Dy(V[y]) + f3 * Dyx(V[y]))
                           + 0.5 * nu * (Df4[y] * Dx(V[x]) + f4 * Dxy(V[x]))
                           + 0.5 * nu * (Df5[y] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f5 * 0.5 * (Dxy(V[y]) + Dyy(V[x])))
                           + 0.5 * nu * (Df6[y] * Dy(V[y]) + f6 * Dyy(V[y]));
            auto Stokes2 = -(Bulk_Dx(P)) + eta * (Dxx(V[y]) + Dyy(V[y]))
                           - 0.5 * nu * (Df1[y] * Dx(V[x]) + f1 * Dxy(V[x]))
                           - 0.5 * nu * (Df2[y] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f2 * 0.5 * (Dxy(V[y]) + Dyy(V[x])))
                           - 0.5 * nu * (Df3[y] * Dy(V[y]) + f3 * Dyy(V[y]))
                           + 0.5 * nu * (Df4[x] * Dx(V[x]) + f4 * Dxx(V[x]))
                           + 0.5 * nu * (Df5[x] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f5 * 0.5 * (Dxx(V[y]) + Dyx(V[x])))
                           + 0.5 * nu * (Df6[x] * Dy(V[y]) + f6 * Dyx(V[y]));

            auto incompressibility=(Dx(V[x]) + Dy(V[y]));

            tt.stop();
            std::cout << "Init of Velocity took " << tt.getwct() << " seconds." << std::endl;
            tt.start();
            RHS[x] = dV[x];
            RHS[y] = dV[y];
            div=-(Dx(dV[x])+Dy(dV[y]));
            Particles.ghost_get<10>(SKIP_LABELLING);
            DCPSE_scheme<equations2d3E, decltype(Particles)> Solver(Particles/*, 41+41+39+39*/);
            Solver.impose(Stokes1, bulk, RHS[0], vx);
            Solver.impose(Stokes2, bulk, RHS[1], vy);
            Solver.impose(V[x], up_p, 0, vx);
            Solver.impose(V[y], up_p, 0, vy);
            Solver.impose(V[x], dw_p, 0, vx);
            Solver.impose(V[y], dw_p, 0, vy);
            Solver.impose(V[x], l_p, 0, vx);
            Solver.impose(V[y], l_p, 0, vy);
            Solver.impose(V[x], r_p, 0, vx);
            Solver.impose(V[y], r_p, 0, vy);
            /*Solver.impose(Helmholtz, bulk, prop_id<19>(), helm);
            Solver.impose(H, up_p1, 0, helm);
            Solver.impose(H, dw_p1, 0, helm);
            Solver.impose(H, l_p1, 0, helm);
            Solver.impose(H, r_p1, 0, helm);
            Solver.impose(-Dx(H) + Dy(H), corner_ul, 0, helm);
            Solver.impose(Dx(H) + Dy(H), corner_ur, 0, helm);
            Solver.impose(-Dx(H) - Dy(H), corner_dl, 0, helm);
            Solver.impose(Dx(H) - Dy(H), corner_dr, 0, helm);
            Solver.impose(P+Dxx(H) + Dyy(H), mid_ref, 0, ic);*/
            Solver.impose(incompressibility, bulk_withoutmid, 0, ic);
            Solver.impose(P, up_p, 0, ic);
            Solver.impose(P, dw_p, 0, ic);
            Solver.impose(P, l_p, 0, ic);
            Solver.impose(P, r_p, 0, ic);
            Solver.impose(P, mid_ref, 0, ic);
            Solver.solve(V[x], V[y], P);
            //Solver.solve(V[x], V[y]);
            Particles.ghost_get<Velocity>(SKIP_LABELLING);
            //SolverH.solve(H);
            //P = (Dxx(H) + Dyy(H));
            for (int i = 0; i < bulk.size(); i++) {
                Particles_subset.getProp<0>(i) = Particles.template getProp<4>(bulk.template get<0>(i));
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<4>(p) = 0;

            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<4>(p) = 0;
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<4>(p) = 0;
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<4>(p) = 0;
            }
            tt.stop();
            std::cout << "Velocity Solved" << std::endl;
            Particles.write("V_DEBUG");
            return;
            u[x][x] = Dx(V[x]);
            u[x][y] = 0.5 * (Dx(V[y]) + Dy(V[x]));
            u[y][x] = 0.5 * (Dy(V[x]) + Dx(V[y]));
            u[y][y] = Dy(V[y]);

            W[x][x] = 0;
            W[x][y] = 0.5 * (Dy(V[x]) - Dx(V[y]));
            W[y][x] = 0.5 * (Dx(V[y]) - Dy(V[x]));
            W[y][y] = 0;

            H_p_b = Pol[x] * Pol[x] + Pol[y] * Pol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<32>(p) = (Particles.getProp<32>(p) == 0) ? 1 : Particles.getProp<32>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<32>(p) = (Particles.getProp<32>(p) == 0) ? 1 : Particles.getProp<32>(p);
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<32>(p) = (Particles.getProp<32>(p) == 0) ? 1 : Particles.getProp<32>(p);
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<32>(p) = (Particles.getProp<32>(p) == 0) ? 1 : Particles.getProp<32>(p);
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<32>(p) = (Particles.getProp<32>(p) == 0) ? 1 : Particles.getProp<32>(p);
            }

            h[x] = -gama * (lambda * delmu - nu * (u[x][x] * Pol[x] * Pol[x] + u[y][y] * Pol[y] * Pol[y] +
                                                   2 * u[x][y] * Pol[x] * Pol[y]) / (H_p_b));


            //Particles.ghost_get<MolField, Strain_rate, Vorticity>(SKIP_LABELLING);
            //Particles.write_frame("Polar_withGhost_3e-3", ctr);
            Particles.deleteGhost();
            Particles.write_frame("Polar_3e-3", ctr);
            Particles.ghost_get<0>();

            ctr++;

            H_p_b = sqrt(H_p_b);

            k1[x] = ((h[x] * Pol[x] - h[y] * Pol[y]) / gama + lambda * delmu * Pol[x] -
                     nu * (u[x][x] * Pol[x] + u[x][y] * Pol[y]) + W[x][x] * Pol[x] +
                     W[x][y] * Pol[y]);// - V[x] * Dx(Pol[x]) - V[y] * Dy(Pol[x]));
            k1[y] = ((h[x] * Pol[y] + h[y] * Pol[x]) / gama + lambda * delmu * Pol[y] -
                     nu * (u[y][x] * Pol[x] + u[y][y] * Pol[y]) + W[y][x] * Pol[x] +
                     W[y][y] * Pol[y]);// - V[x] * Dx(Pol[y]) - V[y] * Dy(Pol[y]));

            H_t = H_p_b;//+0.5*dt*(k1[x]*k1[x]+k1[y]*k1[y]);
            dPol = Pol + (0.5 * dt) * k1;
            dPol = dPol / H_t;
            r = dPol[x] * dPol[x] + dPol[y] * dPol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            Particles.ghost_get<8>(SKIP_LABELLING);


            h[y] = (dPol[x] * (Ks * Dyy(dPol[y]) + Kb * Dxx(dPol[y]) + (Ks - Kb) * Dxy(dPol[x])) -
                    dPol[y] * (Ks * Dxx(dPol[x]) + Kb * Dyy(dPol[x]) + (Ks - Kb) * Dxy(dPol[y])));

            h[x] = -gama * (lambda * delmu - nu * ((u[x][x] * dPol[x] * dPol[x] + u[y][y] * dPol[y] * dPol[y] +
                                                    2 * u[x][y] * dPol[x] * dPol[y]) / (r)));

            k2[x] = ((h[x] * (dPol[x]) - h[y] * (dPol[y])) / gama +
                     lambda * delmu * (dPol[x]) -
                     nu * (u[x][x] * (dPol[x]) + u[x][y] * (dPol[y])) +
                     W[x][x] * (dPol[x]) + W[x][y] * (dPol[y])); //-V[x] * Dx((dPol[x])) - V[y] * Dy((dPol[x])));
            k2[y] = ((h[x] * (dPol[y]) + h[y] * (dPol[x])) / gama +
                     lambda * delmu * (dPol[y]) -
                     nu * (u[y][x] * (dPol[x]) + u[y][y] * (dPol[y])) +
                     W[y][x] * (dPol[x]) + W[y][y] * (dPol[y])); //-V[x] * Dx((dPol[y])) - V[y] * Dy((dPol[y])));

            H_t = H_p_b;//+0.5*dt*(k2[x]*k2[x]+k2[y]*k2[y]);
            dPol = Pol + (0.5 * dt) * k2;
            dPol = dPol / H_t;
            r = dPol[x] * dPol[x] + dPol[y] * dPol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            Particles.ghost_get<8>(SKIP_LABELLING);

            h[y] = (dPol[x] * (Ks * Dyy(dPol[y]) + Kb * Dxx(dPol[y]) + (Ks - Kb) * Dxy(dPol[x])) -
                    dPol[y] * (Ks * Dxx(dPol[x]) + Kb * Dyy(dPol[x]) + (Ks - Kb) * Dxy(dPol[y])));

            h[x] = -gama * (lambda * delmu - nu * ((u[x][x] * dPol[x] * dPol[x] + u[y][y] * dPol[y] * dPol[y] +
                                                    2 * u[x][y] * dPol[x] * dPol[y]) / (r)));

            k3[x] = ((h[x] * (dPol[x]) - h[y] * (dPol[y])) / gama +
                     lambda * delmu * (dPol[x]) -
                     nu * (u[x][x] * (dPol[x]) + u[x][y] * (dPol[y])) +
                     W[x][x] * (dPol[x]) + W[x][y] * (dPol[y]));
            // -V[x] * Dx((dPol[x])) - V[y] * Dy((dPol[x])));
            k3[y] = ((h[x] * (dPol[y]) + h[y] * (dPol[x])) / gama +
                     lambda * delmu * (dPol[y]) -
                     nu * (u[y][x] * (dPol[x]) + u[y][y] * (dPol[y])) +
                     W[y][x] * (dPol[x]) + W[y][y] * (dPol[y]));
            // -V[x] * Dx((dPol[y])) - V[y] * Dy((dPol[y])));
            H_t = H_p_b;//+dt*(k3[x]*k3[x]+k3[y]*k3[y]);
            dPol = Pol + (dt * k3);
            dPol = dPol / H_t;
            Particles.ghost_get<8>(SKIP_LABELLING);
            r = dPol[x] * dPol[x] + dPol[y] * dPol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));

            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            Particles.ghost_get<8>(SKIP_LABELLING);

            h[y] = (dPol[x] * (Ks * Dyy(dPol[y]) + Kb * Dxx(dPol[y]) + (Ks - Kb) * Dxy(dPol[x])) -
                    dPol[y] * (Ks * Dxx(dPol[x]) + Kb * Dyy(dPol[x]) + (Ks - Kb) * Dxy(dPol[y])));

            h[x] = -gama * (lambda * delmu - nu * ((u[x][x] * dPol[x] * dPol[x] + u[y][y] * dPol[y] * dPol[y] +
                                                    2 * u[x][y] * dPol[x] * dPol[y]) / (r)));

            k4[x] = ((h[x] * (dPol[x]) - h[y] * (dPol[y])) / gama +
                     lambda * delmu * (dPol[x]) -
                     nu * (u[x][x] * (dPol[x]) + u[x][y] * (dPol[y])) +
                     W[x][x] * (dPol[x]) +
                     W[x][y] * (dPol[y]));//   -V[x]*Dx( (dt * k3[x]+Pol[x])) -V[y]*Dy( (dt * k3[x]+Pol[x])));
            k4[y] = ((h[x] * (dPol[y]) + h[y] * (dPol[x])) / gama +
                     lambda * delmu * (dPol[y]) -
                     nu * (u[y][x] * (dPol[x]) + u[y][y] * (dPol[y])) +
                     W[y][x] * (dPol[x]) +
                     W[y][y] * (dPol[y]));//  -V[x]*Dx( (dt * k3[y]+Pol*[y])) -V[y]*Dy( (dt * k3[y]+Pol[y])));

            Pol = Pol + (dt / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4);
            Pol = Pol / H_p_b;

            H_p_b = sqrt(Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            Pol = Pol / H_p_b;
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }

            k1 = V;
            k2 = 0.5 * dt * k1 + V;
            k3 = 0.5 * dt * k2 + V;
            k4 = dt * k3 + V;
            //Pos = Pos + dt * V;
            Pos = Pos + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);
            Particles.map();
            Particles.ghost_get<0, ExtForce, 27>();
            indexUpdate(Particles, Particles_subset, up_p, dw_p, l_p, r_p, up_p1, dw_p1, l_p1, r_p1, corner_ul,
                        corner_ur, corner_dl, corner_dr, bulk, up, down, left, right);
            Particles_subset.map();
            Particles_subset.ghost_get<0>();

            //Particles_subset.write("debug");

            tt.start();
            Dx.update(Particles);
            Dy.update(Particles);
            Dxy.update(Particles);
            auto Dyx = Dxy;
            Dxx.update(Particles);
            Dyy.update(Particles);

            Bulk_Dx.update(Particles_subset);
            Bulk_Dy.update(Particles_subset);

            tt.stop();
            std::cout << "Updation of operators took " << tt.getwct() << " seconds." << std::endl;

            std::cout << "Time step " << ctr - 1 << " : " << tim << " over." << std::endl;
            tim += dt;
            std::cout << "----------------------------------------------------------" << std::endl;
        }
        Particles.deleteGhost();
        tt2.stop();
        std::cout << "The simulation took " << tt2.getwct() << "Seconds.";
    }




    BOOST_AUTO_TEST_CASE(Active3D) {
        timer tt2;
        tt2.start();
        double boxsize = 10;
        size_t grd_sz=41;
        const size_t sz[3] = {grd_sz, grd_sz,grd_sz};
        Box<2, double> box({0, 0}, {boxsize, boxsize,boxsize});
        double Lx = box.getHigh(0);
        double Ly = box.getHigh(1);
        double Lz = box.getHigh(2);

        size_t bc[2] = {NON_PERIODIC, NON_PERIODIC,NON_PERIODIC};
        double spacing = box.getHigh(0) / (sz[0] - 1);
        double rCut = 3.9 * spacing;
        double rCut2 = 3.9 * spacing;
        int ord = 2;
        int ord2 = 2;
        double sampling_factor = 4.0;
        double sampling_factor2 = 2.4;
        double alpha_V = 1.0;
        double alpha_P = 1.0;
        Ghost<2, double> ghost(rCut);

        auto &v_cl = create_vcluster();

/*                                          pol                             V         vort                 Ext    Press     strain       stress                      Mfield,   dPol                      dV         RHS                  f1     f2     f3    f4     f5     f6       H               V_t      div    H_t                                                                                      delmu */
        vector_dist<2, double, aggregate<VectorS<3, double>, VectorS<3, double>, double[3][3], VectorS<3, double>, double, double[3][3], double[3][3], VectorS<3, double>, VectorS<3, double>, VectorS<3, double>, VectorS<3, double>, double, double, double, double, double, double, double, VectorS<3, double>, double, double, double[3], double[3], double[3], double[3], double[3], double[3], double, VectorS<3, double>, VectorS<3, double>, VectorS<3, double>, VectorS<3, double>, double, double, double>> Particles(
                0, box, bc, ghost);
        vector_dist<3, double, aggregate<double, double, VectorS<3, double>>> Particles_subset(
                Particles.getDecomposition(), 0);
        double x0, y0, x1, y1;
        x0 = box.getLow(0);
        y0 = box.getLow(1);
        z0 = box.getLow(2);
        x1 = box.getHigh(0);
        y1 = box.getHigh(1);
        z1 = box.getHigh(2);
        auto it = Particles.getGridIterator(sz);
        while (it.isNext()) {
            Particles.add();
            auto key = it.get();
            double x = key.get(0) * it.getSpacing(0);
            Particles.getLastPos()[0] = x;
            double y = key.get(1) * it.getSpacing(1);
            Particles.getLastPos()[1] = y;
            double z = key.get(2) * it.getSpacing(2);
            Particles.getLastPos()[2] = z;
            ++it;
        }

        BOOST_TEST_MESSAGE("Sync domain across processors...");

        Particles.map();
        Particles.ghost_get<0>();

        //Particles.write("Par");

        openfpm::vector<aggregate<int>> bulk;
        openfpm::vector<aggregate<int>> boundary;
        openfpm::vector<aggregate<int>> boundary_without_vertices;
        openfpm::vector<aggregate<int>> f_ul;
        openfpm::vector<aggregate<int>> f_ur;
        openfpm::vector<aggregate<int>> f_dl;
        openfpm::vector<aggregate<int>> f_dr;
        openfpm::vector<aggregate<int>> b_ul;
        openfpm::vector<aggregate<int>> b_ur;
        openfpm::vector<aggregate<int>> b_dl;
        openfpm::vector<aggregate<int>> B_dr;


        constexpr int x = 0;
        constexpr int y = 1;
        constexpr int z = 2;


        constexpr int Polarization = 0;
        constexpr int Velocity = 1;
        constexpr int Vorticity = 2;
        constexpr int ExtForce = 3;
        constexpr int Pressure = 4;
        constexpr int Strain_rate = 5;
        constexpr int Stress = 6;
        constexpr int MolField = 7;
        auto Pos = getV<PROP_POS>(Particles);

        auto Pol = getV<Polarization>(Particles);
        auto V = getV<Velocity>(Particles);
        auto W = getV<Vorticity>(Particles);
        auto g = getV<ExtForce>(Particles);
        auto P = getV<Pressure>(Particles);
        auto P_bulk = getV<0>(Particles_subset); //Pressure only on inside
        auto u = getV<Strain_rate>(Particles);
        auto sigma = getV<Stress>(Particles);
        auto h = getV<MolField>(Particles);
        auto dPol = getV<8>(Particles);
        auto dV = getV<9>(Particles);
        auto RHS = getV<10>(Particles);
        auto f1 = getV<11>(Particles);
        auto f2 = getV<12>(Particles);
        auto f3 = getV<13>(Particles);
        auto f4 = getV<14>(Particles);
        auto f5 = getV<15>(Particles);
        auto f6 = getV<16>(Particles);
        auto H = getV<17>(Particles);
        auto H_bulk = getV<1>(Particles_subset); //Pressure only on inside
        auto Grad_bulk = getV<2>(Particles_subset);

        auto V_t = getV<18>(Particles);
        auto div = getV<19>(Particles);
        auto H_t = getV<20>(Particles);
        auto Df1 = getV<21>(Particles);
        auto Df2 = getV<22>(Particles);
        auto Df3 = getV<23>(Particles);
        auto Df4 = getV<24>(Particles);
        auto Df5 = getV<25>(Particles);
        auto Df6 = getV<26>(Particles);
        auto delmu = getV<27>(Particles);
        auto k1 = getV<28>(Particles);
        auto k2 = getV<29>(Particles);
        auto k3 = getV<30>(Particles);
        auto k4 = getV<31>(Particles);
        auto H_p_b = getV<32>(Particles);
        auto FranckEnergyDensity = getV<33>(Particles);
        auto r = getV<34>(Particles);


        double eta = 1.0;
        double nu = -0.5;
        double gama = 0.1;
        double zeta = 0.07;
        double Ks = 1.0;
        double Kb = 1.0;
        double lambda = 0.1;
        //double delmu = -1.0;
        g = 0;
        delmu = -1.0;
        P = 0;
        P_bulk = 0;
        V = 0;
        // Here fill up the boxes for particle boundary detection.
        Particles.ghost_get<ExtForce, 27>(SKIP_LABELLING);


        Box<3, double> up({box.getLow(0) + spacing / 2.0, box.getHigh(1)- spacing / 2.0,box.getLow(2)+ spacing / 2.0},
                          {box.getHigh(0) - spacing / 2.0, box.getHigh(1) + spacing / 2.0,box.getHigh(2)- spacing / 2.0});

        Box<3, double> down({box.getLow(0) + spacing / 2.0, box.getLow(1) - spacing / 2.0,box.getLow(2)+ spacing / 2.0},
                            {box.getHigh(0) - spacing / 2.0, box.getLow(1) + spacing / 2.0,box.getHigh(2)- spacing / 2.0});

        Box<3, double> left({box.getLow(0) - spacing / 2.0, box.getLow(1) - spacing / 2.0,box.getLow(2) - spacing / 2.0},
                            {box.getLow(0) + spacing / 2.0, box.getHigh(1) + spacing / 2.0,box.getHigh(2) + spacing / 2.0});

        Box<3, double> right({box.getHigh(0) - spacing / 2.0, box.getLow(1) - spacing / 2.0,box.getLow(2)- spacing / 2.0},
                             {box.getHigh(0) + spacing / 2.0, box.getHigh(1) + spacing / 2.0,box.getHigh(2)+ spacing / 2.0});

        Box<3, double> front({box.getLow(0) + spacing / 2.0, box.getLow(1) - spacing / 2.0,box.getLow(2) - spacing / 2.0},
                             {box.getHigh(0) - spacing / 2.0, box.getHigh(1) + spacing / 2.0,box.getLow(2) + spacing / 2.0});

        Box<3, double> back({box.getLow(0) + spacing / 2.0, box.getLow(1) - spacing / 2.0,box.getHigh(2) - spacing / 2.0},
                            {box.getHigh(0) - spacing / 2.0, box.getHigh(1) + spacing / 2.0,box.getHigh(2) + spacing / 2.0});


        openfpm::vector<Box<2, double>> boxes;
        boxes.add(up);
        boxes.add(down);
        boxes.add(left);
        boxes.add(right);


        VTKWriter<openfpm::vector<Box<2, double>>, VECTOR_BOX> vtk_box;
        vtk_box.add(boxes);
        vtk_box.write("vtk_box.vtk");

        auto it2 = Particles.getDomainIterator();

        while (it2.isNext()) {
            auto p = it2.get();
            Point<2, double> xp = Particles.getPos(p);
            Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * xp[x] - Lx) / Lx) - sin((2 * xp[y] - Ly) / Ly)));
            Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * xp[x] - Lx) / Lx) - sin((2 * xp[y] - Ly) / Ly)));
            if (up.isInside(xp) == true) {
                if (left.isInside(xp) == true) {
                    corner_ul.add();
                    corner_ul.last().get<0>() = p.getKey();
                } else if (right.isInside(xp) == true) {
                    corner_ur.add();
                    corner_ur.last().get<0>() = p.getKey();
                } else {
                    up_p1.add();
                    up_p1.last().get<0>() = p.getKey();
                }
                up_p.add();
                up_p.last().get<0>() = p.getKey();

            } else if (down.isInside(xp) == true) {
                if (left.isInside(xp) == true) {
                    corner_dl.add();
                    corner_dl.last().get<0>() = p.getKey();
                } else if (right.isInside(xp) == true) {
                    corner_dr.add();
                    corner_dr.last().get<0>() = p.getKey();
                } else {
                    dw_p1.add();
                    dw_p1.last().get<0>() = p.getKey();
                }
                dw_p.add();
                dw_p.last().get<0>() = p.getKey();
            } else if (left.isInside(xp) == true) {
                if (up.isInside(xp) == false && down.isInside(xp) == false) {
                    l_p1.add();
                    l_p1.last().get<0>() = p.getKey();
                }
                l_p.add();
                l_p.last().get<0>() = p.getKey();
            } else if (right.isInside(xp) == true) {
                if (up.isInside(xp) == false && down.isInside(xp) == false) {
                    r_p1.add();
                    r_p1.last().get<0>() = p.getKey();
                }
                r_p.add();
                r_p.last().get<0>() = p.getKey();
            } else {
                bulk.add();
                bulk.last().get<0>() = p.getKey();
            }
            ++it2;
        }


        for (int i = 0; i < bulk.size(); i++) {
            Particles_subset.add();
            Particles_subset.getLastPos()[0] = Particles.getPos(bulk.template get<0>(i))[0];
            Particles_subset.getLastPos()[1] = Particles.getPos(bulk.template get<0>(i))[1];
        }


        Particles_subset.map();
        Particles_subset.ghost_get<0>();

        //Particles_subset.write("Pars");
        Derivative_x Dx(Particles, ord, rCut, sampling_factor, support_options::RADIUS), Bulk_Dx(Particles_subset, ord,
                                                                                                 rCut,
                                                                                                 sampling_factor,
                                                                                                 support_options::RADIUS);
        Derivative_y Dy(Particles, ord, rCut, sampling_factor, support_options::RADIUS), Bulk_Dy(Particles_subset, ord,
                                                                                                 rCut,
                                                                                                 sampling_factor,
                                                                                                 support_options::RADIUS);
        Derivative_xy Dxy(Particles, ord, rCut2, sampling_factor2, support_options::RADIUS);
        auto Dyx = Dxy;
        Derivative_xx Dxx(Particles, ord, rCut2, sampling_factor2,
                          support_options::RADIUS);//, Dxx2(Particles, ord2, rCut2, sampling_factor2, support_options::RADIUS);
        Derivative_yy Dyy(Particles, ord, rCut2, sampling_factor2,
                          support_options::RADIUS);//, Dyy2(Particles, ord2, rCut2, sampling_factor2, support_options::RADIUS);

/*        Derivative_x Dx(Particles, ord, rCut, sampling_factor), Bulk_Dx(Particles_subset, ord, rCut, sampling_factor);
        Derivative_y Dy(Particles, ord, rCut, sampling_factor), Bulk_Dy(Particles_subset, ord, rCut, sampling_factor);
        Derivative_xy Dxy(Particles, ord2, rCut, sampling_factor);
        auto Dyx = Dxy;
        Derivative_xx Dxx(Particles, ord2, rCut2, sampling_factor2),Bulk_Dxx(Particles_subset, ord2, rCut2, sampling_factor2);
        Derivative_yy Dyy(Particles, ord2, rCut2, sampling_factor2),Bulk_Dyy(Particles_subset, ord2, rCut2, sampling_factor2);*/




        eq_id vx, vy;
        timer tt;
        timer tt3;
        vx.setId(0);
        vy.setId(1);
        double V_err_eps = 5 * 1e-4;
        double V_err = 1, V_err_old;
        int n = 0;
        int nmax = 300;
        int ctr = 0, errctr, Vreset = 0;
        double dt = 2e-7;

        double tim = 0;
        double tf = 2e-6;
        div = 0;
        double sum, sum1, sum_k;
        while (tim <= tf) {
            tt.start();
            petsc_solver<double> solverPetsc;
            solverPetsc.setSolver(KSPGMRES);
            //solverPetsc.setRestart(250);
            solverPetsc.setPreconditioner(PCJACOBI);
            petsc_solver<double> solverPetsc2;
            solverPetsc2.setSolver(KSPGMRES);
            solverPetsc2.setPreconditioner(PCJACOBI);

            Particles.ghost_get<Polarization>(SKIP_LABELLING);
            sigma[x][x] =
                    -Ks * Dx(Pol[x]) * Dx(Pol[x]) - Kb * Dx(Pol[y]) * Dx(Pol[y]) + (Kb - Ks) * Dy(Pol[x]) * Dx(Pol[y]);
            sigma[x][y] =
                    -Ks * Dy(Pol[y]) * Dx(Pol[y]) - Kb * Dy(Pol[x]) * Dx(Pol[x]) + (Kb - Ks) * Dx(Pol[y]) * Dx(Pol[x]);
            sigma[y][x] =
                    -Ks * Dx(Pol[x]) * Dy(Pol[x]) - Kb * Dx(Pol[y]) * Dy(Pol[y]) + (Kb - Ks) * Dy(Pol[x]) * Dy(Pol[y]);
            sigma[y][y] =
                    -Ks * Dy(Pol[y]) * Dy(Pol[y]) - Kb * Dy(Pol[x]) * Dy(Pol[x]) + (Kb - Ks) * Dx(Pol[y]) * Dy(Pol[x]);
            Particles.ghost_get<Stress>(SKIP_LABELLING);


            r = Pol[x] * Pol[x] + Pol[y] * Pol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
            }

            h[y] = (Pol[x] * (Ks * Dyy(Pol[y]) + Kb * Dxx(Pol[y]) + (Ks - Kb) * Dxy(Pol[x])) -
                    Pol[y] * (Ks * Dxx(Pol[x]) + Kb * Dyy(Pol[x]) + (Ks - Kb) * Dxy(Pol[y])));

            Particles.ghost_get<MolField>(SKIP_LABELLING);

            FranckEnergyDensity = (Ks / 2.0) *
                                  ((Dx(Pol[x]) * Dx(Pol[x])) + (Dy(Pol[x]) * Dy(Pol[x])) +
                                   (Dx(Pol[y]) * Dx(Pol[y])) +
                                   (Dy(Pol[y]) * Dy(Pol[y]))) +
                                  ((Kb - Ks) / 2.0) * ((Dx(Pol[y]) - Dy(Pol[x])) * (Dx(Pol[y]) - Dy(Pol[x])));
            Particles.ghost_get<33>(SKIP_LABELLING);


            f1 = gama * nu * Pol[x] * Pol[x] * (Pol[x] * Pol[x] - Pol[y] * Pol[y]) / (r);
            f2 = 2.0 * gama * nu * Pol[x] * Pol[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y]) / (r);
            f3 = gama * nu * Pol[y] * Pol[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y]) / (r);
            f4 = 2.0 * gama * nu * Pol[x] * Pol[x] * Pol[x] * Pol[y] / (r);
            f5 = 4.0 * gama * nu * Pol[x] * Pol[x] * Pol[y] * Pol[y] / (r);
            f6 = 2.0 * gama * nu * Pol[x] * Pol[y] * Pol[y] * Pol[y] / (r);
            Particles.ghost_get<11, 12, 13, 14, 15, 16>(SKIP_LABELLING);
            Df1[x] = Dx(f1);
            Df2[x] = Dx(f2);
            Df3[x] = Dx(f3);
            Df4[x] = Dx(f4);
            Df5[x] = Dx(f5);
            Df6[x] = Dx(f6);

            Df1[y] = Dy(f1);
            Df2[y] = Dy(f2);
            Df3[y] = Dy(f3);
            Df4[y] = Dy(f4);
            Df5[y] = Dy(f5);
            Df6[y] = Dy(f6);
            Particles.ghost_get<21, 22, 23, 24, 25, 26>(SKIP_LABELLING);


            dV[x] = -0.5 * Dy(h[y]) + zeta * Dx(delmu * Pol[x] * Pol[x]) + zeta * Dy(delmu * Pol[x] * Pol[y]) -
                    zeta * Dx(0.5 * delmu * (Pol[x] * Pol[x] + Pol[y] * Pol[y])) -
                    0.5 * nu * Dx(-2.0 * h[y] * Pol[x] * Pol[y])
                    - 0.5 * nu * Dy(h[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y])) - Dx(sigma[x][x]) -
                    Dy(sigma[x][y]) -
                    g[x]
                    - 0.5 * nu * Dx(-gama * lambda * delmu * (Pol[x] * Pol[x] - Pol[y] * Pol[y]))
                    - 0.5 * Dy(-2.0 * gama * lambda * delmu * (Pol[x] * Pol[y]));


            dV[y] = -0.5 * Dx(-h[y]) + zeta * Dy(delmu * Pol[y] * Pol[y]) + zeta * Dx(delmu * Pol[x] * Pol[y]) -
                    zeta * Dy(0.5 * delmu * (Pol[x] * Pol[x] + Pol[y] * Pol[y])) -
                    0.5 * nu * Dy(2.0 * h[y] * Pol[x] * Pol[y])
                    - 0.5 * nu * Dx(h[y] * (Pol[x] * Pol[x] - Pol[y] * Pol[y])) - Dx(sigma[y][x]) -
                    Dy(sigma[y][y]) -
                    g[y]
                    - 0.5 * nu * Dy(gama * lambda * delmu * (Pol[x] * Pol[x] - Pol[y] * Pol[y]))
                    - 0.5 * Dx(-2.0 * gama * lambda * delmu * (Pol[x] * Pol[y]));
            Particles.ghost_get<9>(SKIP_LABELLING);


            //Particles.write("PolarI");
            //Velocity Solution n iterations


            auto Stokes1 = eta * (Dxx(V[x]) + Dyy(V[x]))
                           + 0.5 * nu * (Df1[x] * Dx(V[x]) + f1 * Dxx(V[x]))
                           + 0.5 * nu * (Df2[x] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f2 * 0.5 * (Dxx(V[y]) + Dyx(V[x])))
                           + 0.5 * nu * (Df3[x] * Dy(V[y]) + f3 * Dyx(V[y]))
                           + 0.5 * nu * (Df4[y] * Dx(V[x]) + f4 * Dxy(V[x]))
                           + 0.5 * nu * (Df5[y] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f5 * 0.5 * (Dxy(V[y]) + Dyy(V[x])))
                           + 0.5 * nu * (Df6[y] * Dy(V[y]) + f6 * Dyy(V[y]));
            auto Stokes2 = eta * (Dxx(V[y]) + Dyy(V[y]))
                           - 0.5 * nu * (Df1[y] * Dx(V[x]) + f1 * Dxy(V[x]))
                           - 0.5 * nu * (Df2[y] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f2 * 0.5 * (Dxy(V[y]) + Dyy(V[x])))
                           - 0.5 * nu * (Df3[y] * Dy(V[y]) + f3 * Dyy(V[y]))
                           + 0.5 * nu * (Df4[x] * Dx(V[x]) + f4 * Dxx(V[x]))
                           + 0.5 * nu * (Df5[x] * 0.5 * (Dx(V[y]) + Dy(V[x])) + f5 * 0.5 * (Dxx(V[y]) + Dyx(V[x])))
                           + 0.5 * nu * (Df6[x] * Dy(V[y]) + f6 * Dyx(V[y]));
            tt.stop();
            std::cout << "Init of Velocity took " << tt.getwct() << " seconds." << std::endl;
            tt.start();
            V_err = 1;
            n = 0;
            errctr = 0;
            if (Vreset == 1) {
                P_bulk = 0;
                P = 0;
                Vreset = 0;
            }
            P = 0;
            P_bulk = 0;

            //////////////////////// DEBUG test //////////////////////////

//            DCPSE_scheme<equations2d2, decltype(Particles)> Solver(Particles);
            /*PLAN FOR HACKATHON


            P=Dx(V);
            Pbulk=Dx(V);
            Pbulk=Bulk_Dx(V);
             for a tensor say K= double[2][2], K[0]
             temp = Pol+Pol
             */

//            DCPSE_scheme<equations2d1, decltype(Particles)> SolverH(Particles);

            //////////////////////////////////////////////////////////////

            while (V_err >= V_err_eps && n <= nmax) {
                RHS[x] = dV[x];
                RHS[y] = dV[y];
                Particles_subset.ghost_get<0>(SKIP_LABELLING);
                Grad_bulk[x] = Bulk_Dx(P_bulk);
                Grad_bulk[y] = Bulk_Dy(P_bulk);
                for (int i = 0; i < bulk.size(); i++) {
                    Particles.template getProp<10>(bulk.template get<0>(i))[x] += Particles_subset.getProp<2>(i)[x];
                    Particles.template getProp<10>(bulk.template get<0>(i))[y] += Particles_subset.getProp<2>(i)[y];
                }
                Particles.ghost_get<10>(SKIP_LABELLING);
                DCPSE_scheme<equations2d2, decltype(Particles)> Solver(Particles);
//                Solver.reset(Particles);
                Solver.impose(Stokes1, bulk, RHS[0], vx);
                Solver.impose(Stokes2, bulk, RHS[1], vy);
                Solver.impose(V[x], up_p, 0, vx);
                Solver.impose(V[y], up_p, 0, vy);
                Solver.impose(V[x], dw_p, 0, vx);
                Solver.impose(V[y], dw_p, 0, vy);
                Solver.impose(V[x], l_p, 0, vx);
                Solver.impose(V[y], l_p, 0, vy);
                Solver.impose(V[x], r_p, 0, vx);
                Solver.impose(V[y], r_p, 0, vy);
                Solver.solve_with_solver(solverPetsc, V[x], V[y]);
                //Solver.solve(V[x], V[y]);
                Particles.ghost_get<Velocity>(SKIP_LABELLING);
                div = -(Dx(V[x]) + Dy(V[y]));
                auto Helmholtz = Dxx(H) + Dyy(H);
                DCPSE_scheme<equations2d1, decltype(Particles)> SolverH(Particles);
//                SolverH.reset(Particles);
                SolverH.impose(Helmholtz, bulk, prop_id<19>());
                SolverH.impose(H, up_p1, 0);
                SolverH.impose(H, dw_p1, 0);
                SolverH.impose(H, l_p1, 0);
                SolverH.impose(H, r_p1, 0);
                SolverH.impose(-Dx(H) + Dy(H), corner_ul, 0);
                SolverH.impose(Dx(H) + Dy(H), corner_ur, 0);
                SolverH.impose(-Dx(H) - Dy(H), corner_dl, 0);
                SolverH.impose(Dx(H) - Dy(H), corner_dr, 0);
                SolverH.solve_with_solver(solverPetsc2, H);
                //SolverH.solve(H);
                P = P + div;
                for (int i = 0; i < bulk.size(); i++) {
                    Particles_subset.getProp<0>(i) = Particles.template getProp<4>(bulk.template get<0>(i));
                }
                for (int j = 0; j < up_p.size(); j++) {
                    auto p = up_p.get<0>(j);
                    Particles.getProp<4>(p) = 0;

                }
                for (int j = 0; j < dw_p.size(); j++) {
                    auto p = dw_p.get<0>(j);
                    Particles.getProp<4>(p) = 0;
                }
                for (int j = 0; j < l_p.size(); j++) {
                    auto p = l_p.get<0>(j);
                    Particles.getProp<4>(p) = 0;
                }
                for (int j = 0; j < r_p.size(); j++) {
                    auto p = r_p.get<0>(j);
                    Particles.getProp<4>(p) = 0;
                }
                sum = 0;
                sum1 = 0;
                for (int j = 0; j < bulk.size(); j++) {
                    auto p = bulk.get<0>(j);
                    sum += (Particles.getProp<18>(p)[0] - Particles.getProp<1>(p)[0]) *
                           (Particles.getProp<18>(p)[0] - Particles.getProp<1>(p)[0]) +
                           (Particles.getProp<18>(p)[1] - Particles.getProp<1>(p)[1]) *
                           (Particles.getProp<18>(p)[1] - Particles.getProp<1>(p)[1]);
                    sum1 += Particles.getProp<1>(p)[0] * Particles.getProp<1>(p)[0] +
                            Particles.getProp<1>(p)[1] * Particles.getProp<1>(p)[1];
                }
                sum = sqrt(sum);
                sum1 = sqrt(sum1);

                v_cl.sum(sum);
                v_cl.sum(sum1);
                v_cl.execute();
                V_t = V;
                Particles.ghost_get<1,4,18>(SKIP_LABELLING);
                V_err_old = V_err;
                V_err = sum / sum1;
                if (V_err > V_err_old || abs(V_err_old - V_err) < 1e-8) {
                    errctr++;
                    //alpha_P -= 0.1;
                } else {
                    errctr = 0;
                }
                if (n > 3) {
                    if (errctr > 3) {
                        std::cout << "CONVERGENCE LOOP BROKEN DUE TO INCREASE/VERY SLOW DECREASE IN ERROR" << std::endl;
                        Vreset = 1;
                        break;
                    } else {
                        Vreset = 0;
                    }
                }
                n++;
                //Particles.write_frame("V_debug", n);
                if (v_cl.rank() == 0) {
                    std::cout << "Rel l2 cgs err in V = " << V_err << " at " << n << std::endl;
                }
            }
            tt.stop();
            u[x][x] = Dx(V[x]);
            u[x][y] = 0.5 * (Dx(V[y]) + Dy(V[x]));
            u[y][x] = 0.5 * (Dy(V[x]) + Dx(V[y]));
            u[y][y] = Dy(V[y]);


            //Adaptive CFL
            /*sum=0;
            auto it2 = Particles.getDomainIterator();
            while (it2.isNext()) {
                auto p = it2.get();
                sum += Particles.getProp<Strain_rate>(p)[x][x] * Particles.getProp<Strain_rate>(p)[x][x] +
                        Particles.getProp<Strain_rate>(p)[y][y] * Particles.getProp<Strain_rate>(p)[y][y];
                ++it2;
            }
            sum = sqrt(sum);
            v_cl.sum(sum);
            v_cl.execute();
            dt=0.5/sum;*/
            std::cout << "Rel l2 cgs err in V = " << V_err << " and took " << tt.getwct() << " seconds with " << n
                      << " iterations. dt is set to "<<dt
                      << std::endl;

            W[x][x] = 0;
            W[x][y] = 0.5 * (Dy(V[x]) - Dx(V[y]));
            W[y][x] = 0.5 * (Dx(V[y]) - Dy(V[x]));
            W[y][y] = 0;

            H_p_b = Pol[x] * Pol[x] + Pol[y] * Pol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<32>(p) = (Particles.getProp<32>(p) == 0) ? 1 : Particles.getProp<32>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<32>(p) = (Particles.getProp<32>(p) == 0) ? 1 : Particles.getProp<32>(p);
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<32>(p) = (Particles.getProp<32>(p) == 0) ? 1 : Particles.getProp<32>(p);
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<32>(p) = (Particles.getProp<32>(p) == 0) ? 1 : Particles.getProp<32>(p);
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<32>(p) = (Particles.getProp<32>(p) == 0) ? 1 : Particles.getProp<32>(p);
            }

            h[x] = -gama * (lambda * delmu - nu * (u[x][x] * Pol[x] * Pol[x] + u[y][y] * Pol[y] * Pol[y] +
                                                   2 * u[x][y] * Pol[x] * Pol[y]) / (H_p_b));


            //Particles.ghost_get<MolField, Strain_rate, Vorticity>(SKIP_LABELLING);
            //Particles.write_frame("Polar_withGhost_3e-3", ctr);
            Particles.deleteGhost();
            Particles.write_frame("Polar_2e-7", ctr);
            Particles.ghost_get<0>();

            ctr++;

            H_p_b = sqrt(H_p_b);

            k1[x] = ((h[x] * Pol[x] - h[y] * Pol[y]) / gama + lambda * delmu * Pol[x] -
                     nu * (u[x][x] * Pol[x] + u[x][y] * Pol[y]) + W[x][x] * Pol[x] +
                     W[x][y] * Pol[y]);// - V[x] * Dx(Pol[x]) - V[y] * Dy(Pol[x]));
            k1[y] = ((h[x] * Pol[y] + h[y] * Pol[x]) / gama + lambda * delmu * Pol[y] -
                     nu * (u[y][x] * Pol[x] + u[y][y] * Pol[y]) + W[y][x] * Pol[x] +
                     W[y][y] * Pol[y]);// - V[x] * Dx(Pol[y]) - V[y] * Dy(Pol[y]));

            H_t = H_p_b;//+0.5*dt*(k1[x]*k1[x]+k1[y]*k1[y]);
            dPol = Pol + (0.5 * dt) * k1;
            dPol = dPol / H_t;
            r = dPol[x] * dPol[x] + dPol[y] * dPol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            Particles.ghost_get<8>(SKIP_LABELLING);


            h[y] = (dPol[x] * (Ks * Dyy(dPol[y]) + Kb * Dxx(dPol[y]) + (Ks - Kb) * Dxy(dPol[x])) -
                    dPol[y] * (Ks * Dxx(dPol[x]) + Kb * Dyy(dPol[x]) + (Ks - Kb) * Dxy(dPol[y])));

            h[x] = -gama * (lambda * delmu - nu * ((u[x][x] * dPol[x] * dPol[x] + u[y][y] * dPol[y] * dPol[y] +
                                                    2 * u[x][y] * dPol[x] * dPol[y]) / (r)));

            k2[x] = ((h[x] * (dPol[x]) - h[y] * (dPol[y])) / gama +
                     lambda * delmu * (dPol[x]) -
                     nu * (u[x][x] * (dPol[x]) + u[x][y] * (dPol[y])) +
                     W[x][x] * (dPol[x]) + W[x][y] * (dPol[y])); //-V[x] * Dx((dPol[x])) - V[y] * Dy((dPol[x])));
            k2[y] = ((h[x] * (dPol[y]) + h[y] * (dPol[x])) / gama +
                     lambda * delmu * (dPol[y]) -
                     nu * (u[y][x] * (dPol[x]) + u[y][y] * (dPol[y])) +
                     W[y][x] * (dPol[x]) + W[y][y] * (dPol[y])); //-V[x] * Dx((dPol[y])) - V[y] * Dy((dPol[y])));

            H_t = H_p_b;//+0.5*dt*(k2[x]*k2[x]+k2[y]*k2[y]);
            dPol = Pol + (0.5 * dt) * k2;
            dPol = dPol / H_t;
            r = dPol[x] * dPol[x] + dPol[y] * dPol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            Particles.ghost_get<8>(SKIP_LABELLING);

            h[y] = (dPol[x] * (Ks * Dyy(dPol[y]) + Kb * Dxx(dPol[y]) + (Ks - Kb) * Dxy(dPol[x])) -
                    dPol[y] * (Ks * Dxx(dPol[x]) + Kb * Dyy(dPol[x]) + (Ks - Kb) * Dxy(dPol[y])));

            h[x] = -gama * (lambda * delmu - nu * ((u[x][x] * dPol[x] * dPol[x] + u[y][y] * dPol[y] * dPol[y] +
                                                    2 * u[x][y] * dPol[x] * dPol[y]) / (r)));

            k3[x] = ((h[x] * (dPol[x]) - h[y] * (dPol[y])) / gama +
                     lambda * delmu * (dPol[x]) -
                     nu * (u[x][x] * (dPol[x]) + u[x][y] * (dPol[y])) +
                     W[x][x] * (dPol[x]) + W[x][y] * (dPol[y]));
            // -V[x] * Dx((dPol[x])) - V[y] * Dy((dPol[x])));
            k3[y] = ((h[x] * (dPol[y]) + h[y] * (dPol[x])) / gama +
                     lambda * delmu * (dPol[y]) -
                     nu * (u[y][x] * (dPol[x]) + u[y][y] * (dPol[y])) +
                     W[y][x] * (dPol[x]) + W[y][y] * (dPol[y]));
            // -V[x] * Dx((dPol[y])) - V[y] * Dy((dPol[y])));
            H_t = H_p_b;//+dt*(k3[x]*k3[x]+k3[y]*k3[y]);
            dPol = Pol + (dt * k3);
            dPol = dPol / H_t;
            Particles.ghost_get<8>(SKIP_LABELLING);
            r = dPol[x] * dPol[x] + dPol[y] * dPol[y];
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
            }
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));

            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<34>(p) = (Particles.getProp<34>(p) == 0) ? 1 : Particles.getProp<34>(p);
                Particles.getProp<8>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            Particles.ghost_get<8>(SKIP_LABELLING);

            h[y] = (dPol[x] * (Ks * Dyy(dPol[y]) + Kb * Dxx(dPol[y]) + (Ks - Kb) * Dxy(dPol[x])) -
                    dPol[y] * (Ks * Dxx(dPol[x]) + Kb * Dyy(dPol[x]) + (Ks - Kb) * Dxy(dPol[y])));

            h[x] = -gama * (lambda * delmu - nu * ((u[x][x] * dPol[x] * dPol[x] + u[y][y] * dPol[y] * dPol[y] +
                                                    2 * u[x][y] * dPol[x] * dPol[y]) / (r)));

            k4[x] = ((h[x] * (dPol[x]) - h[y] * (dPol[y])) / gama +
                     lambda * delmu * (dPol[x]) -
                     nu * (u[x][x] * (dPol[x]) + u[x][y] * (dPol[y])) +
                     W[x][x] * (dPol[x]) +
                     W[x][y] * (dPol[y]));//   -V[x]*Dx( (dt * k3[x]+Pol[x])) -V[y]*Dy( (dt * k3[x]+Pol[x])));
            k4[y] = ((h[x] * (dPol[y]) + h[y] * (dPol[x])) / gama +
                     lambda * delmu * (dPol[y]) -
                     nu * (u[y][x] * (dPol[x]) + u[y][y] * (dPol[y])) +
                     W[y][x] * (dPol[x]) +
                     W[y][y] * (dPol[y]));//  -V[x]*Dx( (dt * k3[y]+Pol*[y])) -V[y]*Dy( (dt * k3[y]+Pol[y])));

            Pol = Pol + (dt / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4);
            Pol = Pol / H_p_b;

            H_p_b = sqrt(Pol[x] * Pol[x] + Pol[y] * Pol[y]);
            Pol = Pol / H_p_b;
            for (int j = 0; j < up_p.size(); j++) {
                auto p = up_p.get<0>(j);
                Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < dw_p.size(); j++) {
                auto p = dw_p.get<0>(j);
                Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < l_p.size(); j++) {
                auto p = l_p.get<0>(j);
                Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }
            for (int j = 0; j < r_p.size(); j++) {
                auto p = r_p.get<0>(j);
                Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            }

            k1 = V;
            k2 = 0.5 * dt * k1 + V;
            k3 = 0.5 * dt * k2 + V;
            k4 = dt * k3 + V;
            //Pos = Pos + dt * V;
            Pos = Pos + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);

            Particles.map();

            Particles.ghost_get<0, ExtForce, 27>();
            indexUpdate(Particles, Particles_subset, up_p, dw_p, l_p, r_p, up_p1, dw_p1, l_p1, r_p1, corner_ul,
                        corner_ur, corner_dl, corner_dr, bulk, up, down, left, right);
            Particles_subset.map();
            Particles_subset.ghost_get<0>();

            //Particles_subset.write("debug");

            tt.start();
            Dx.update(Particles);
            Dy.update(Particles);
            Dxy.update(Particles);
            auto Dyx = Dxy;
            Dxx.update(Particles);
            Dyy.update(Particles);

            Bulk_Dx.update(Particles_subset);
            Bulk_Dy.update(Particles_subset);

            tt.stop();
            std::cout << "Updation of operators took " << tt.getwct() << " seconds." << std::endl;

            std::cout << "Time step " << ctr - 1 << " : " << tim << " over." << std::endl;
            tim += dt;
            std::cout << "----------------------------------------------------------" << std::endl;
        }

        Dx.deallocate(Particles);
        Dy.deallocate(Particles);
        Dxy.deallocate(Particles);
        Dxx.deallocate(Particles);
        Dyy.deallocate(Particles);
        Bulk_Dx.deallocate(Particles_subset);
        Bulk_Dy.deallocate(Particles_subset);
        Particles.deleteGhost();
        tt2.stop();
        std::cout << "The simulation took " << tt2.getwct() << "Seconds.";
    }
















BOOST_AUTO_TEST_SUITE_END()
/*
// Add multi res patch 1
{
const size_t sz2[2] = {81,81};

//Box<2,double> bx({0.25 + it.getSpacing(0)/4.0,0.25 + it.getSpacing(0)/4.0},{sz2[0]*it.getSpacing(0)/2.0 + 0.25 + it.getSpacing(0)/4.0, sz2[1]*it.getSpacing(0)/2.0 + 0.25 + it.getSpacing(0)/4.0});
Box<2, double> bx({box.getLow(0) + 10.1 * spacing, box.getLow(1) + 10.1 * spacing},
                  {box.getHigh(0) - 10.1 * spacing, box.getHigh(1) - 10.1 * spacing});
openfpm::vector<size_t> rem;

auto it = Particles.getDomainIterator();

while (it.isNext())
{
auto k = it.get();

Point<2,double> xp = Particles.getPos(k);

if (bx.isInside(xp) == true)
{
rem.add(k.getKey());
}

++it;
}

Particles.remove(rem);
Box<2, double> bx2({box.getLow(0) + 14.1 * spacing, box.getLow(1) + 14.1 * spacing},
                   {box.getHigh(0) - 14.1 * spacing, box.getHigh(1) - 14.1 * spacing});
double spacing2 = bx2.getHigh(0) / (sz2[0] - 1);
auto it2 = Particles.getGridIterator(sz2);
while (it2.isNext()) {
auto key = it2.get();
double x = key.get(0) * spacing2 + 14.1 * spacing;
double y = key.get(1) * spacing2 + 14.1 * spacing;
Point<2,double> xp ={x,y};
if (bx2.isInside(xp) == true)
{
Particles.add();
Particles.getLastPos()[0] = x;
Particles.getLastPos()[1] = y;
}
++it2;
}
}*/


while (it2.isNext()) {
auto p = it2.get();
Point<3, double> xp = Particles.getPos(p);
Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * xp[x] - Lx) / Lx) - sin((2 * xp[y] - Ly) / Ly)));
Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * xp[x] - Lx) / Lx) - sin((2 * xp[y] - Ly) / Ly)));
Particles.getProp<0>(p)[z] = 0;
if (front.isInside(xp) == true) {
if (up.isInside(xp) == true) {
if (left.isInside(xp) == true) {
f_ul.add();
f_ul.last().get<0>() = p.getKey();
} else if (right.isInside(xp) == true) {
f_ur.add();
f_ur.last().get<0>() = p.getKey();
} else {
Surface_without_corners.add();
Surface_without_corners.last().get<0>() = p.getKey();
}
} else if (down.isInside(xp) == true) {
if (left.isInside(xp) == true) {
f_dl.add();
f_dl.last().get<0>() = p.getKey();
}
if (right.isInside(xp) == true) {
f_dr.add();
f_dr.last().get<0>() = p.getKey();
} else {
Surface_without_corners.add();
Surface_without_corners.last().get<0>() = p.getKey();
}
}
Surface.add();
Surface.last().get<0>() = p.getKey();
} else if (back.isInside(xp) == true) {
if (up.isInside(xp) == true) {
if (left.isInside(xp) == true) {
b_ul.add();
b_ul.last().get<0>() = p.getKey();
} else if (right.isInside(xp) == true) {
b_ur.add();
b_ur.last().get<0>() = p.getKey();
} else {
Surface_without_corners.add();
Surface_without_corners.last().get<0>() = p.getKey();
}
} else if (down.isInside(xp) == true) {
if (left.isInside(xp) == true) {
b_dl.add();
b_dl.last().get<0>() = p.getKey();
}
if (right.isInside(xp) == true) {
b_dr.add();
b_dr.last().get<0>() = p.getKey();
} else {
Surface_without_corners.add();
Surface_without_corners.last().get<0>() = p.getKey();
}
}
Surface.add();
Surface.last().get<0>() = p.getKey();
} else if (left.isInside(xp) == true) {
Surface_without_corners.add();
Surface_without_corners.last().get<0>() = p.getKey();
Surface.add();
Surface.last().get<0>() = p.getKey();
} else if (right.isInside(xp) == true) {
Surface_without_corners.add();
Surface_without_corners.last().get<0>() = p.getKey();
Surface.add();
Surface.last().get<0>() = p.getKey();
} else if (up.isInside(xp) == true) {
Surface_without_corners.add();
Surface_without_corners.last().get<0>() = p.getKey();
Surface.add();
Surface.last().get<0>() = p.getKey();
} else if (down.isInside(xp) == true) {
Surface_without_corners.add();
Surface_without_corners.last().get<0>() = p.getKey();
Surface.add();
Surface.last().get<0>() = p.getKey();
} else {
bulk.add();
bulk.last().get<0>() = p.getKey();
}
++it2;
}
BOOST_AUTO_TEST_CASE(Active3dSimple) {
    timer tt2;
    tt2.start();
    size_t grd_sz = 21;
    double dt = 1e-3;
    double boxsize = 10;
    const size_t sz[3] = {grd_sz, grd_sz, grd_sz};
    Box<3, double> box({0, 0, 0}, {boxsize, boxsize, boxsize});
    size_t bc[3] = {NON_PERIODIC, NON_PERIODIC, NON_PERIODIC};
    double Lx = box.getHigh(0);
    double Ly = box.getHigh(1);
    double Lz = box.getHigh(2);
    double spacing = box.getHigh(0) / (sz[0] - 1);
    double rCut = 3.9 * spacing;
    double rCut2 = 3.9 * spacing;
    int ord = 2;
    int ord2 = 2;
    double sampling_factor = 4.0;
    double sampling_factor2 = 2.4;
    Ghost<3, double> ghost(rCut);
    auto &v_cl = create_vcluster();
    /*                                 pol                          V           vort              Ext                 Press     strain       stress                      Mfield,   dPol                      dV         RHS                  f1     f2     f3    f4     f5     f6       H               V_t      div    H_t                                                                                      delmu */
    vector_dist<3, double, aggregate<VectorS<3, double>, VectorS<3, double>, double[3][3], VectorS<3, double>, double, double[3][3], double[3][3], VectorS<3, double>, VectorS<3, double>, VectorS<3, double>, VectorS<3, double>, double, double, double, double, double, double, double, VectorS<3, double>, double, double, double[3], double[3], double[3], double[3], double[3], double[3], double, VectorS<3, double>, VectorS<3, double>, VectorS<3, double>, VectorS<3, double>, double, double, double, double[3][3],double[3][3]>> Particles(
            0, box, bc, ghost);


    double x0, y0, z0, x1, y1, z1;
    x0 = box.getLow(0);
    y0 = box.getLow(1);
    z0 = box.getLow(2);
    x1 = box.getHigh(0);
    y1 = box.getHigh(1);
    z1 = box.getHigh(2);

    auto it = Particles.getGridIterator(sz);
    while (it.isNext()) {
        Particles.add();
        auto key = it.get();
        double x = key.get(0) * it.getSpacing(0);
        Particles.getLastPos()[0] = x;
        double y = key.get(1) * it.getSpacing(1);
        Particles.getLastPos()[1] = y;
        double z = key.get(2) * it.getSpacing(1);
        Particles.getLastPos()[2] = z;
        ++it;
    }

    Particles.map();
    Particles.ghost_get<0>();

    openfpm::vector<aggregate<int>> bulk;
    openfpm::vector<aggregate<int>> Boundary;

    constexpr int x = 0;
    constexpr int y = 1;
    constexpr int z = 2;


    constexpr int Polarization = 0;
    constexpr int Velocity = 1;
    constexpr int Vorticity = 2;
    constexpr int ExtForce = 3;
    constexpr int Pressure = 4;
    constexpr int Strain_rate = 5;
    constexpr int Stress = 6;
    constexpr int MolField = 7;
    auto Pos = getV<PROP_POS>(Particles);

    auto Pol = getV<Polarization>(Particles);
    auto V = getV<Velocity>(Particles);
    auto W = getV<Vorticity>(Particles);
    auto g = getV<ExtForce>(Particles);
    auto P = getV<Pressure>(Particles);
    auto u = getV<Strain_rate>(Particles);
    auto sigma = getV<Stress>(Particles);
    auto h = getV<MolField>(Particles);
    auto dPol = getV<8>(Particles);
    auto dV = getV<9>(Particles);
    auto RHS = getV<10>(Particles);
    auto f1 = getV<11>(Particles);
    auto f2 = getV<12>(Particles);
    auto f3 = getV<13>(Particles);
    auto f4 = getV<14>(Particles);
    auto f5 = getV<15>(Particles);
    auto f6 = getV<16>(Particles);
    auto H = getV<17>(Particles);
    auto V_t = getV<18>(Particles);
    auto div = getV<19>(Particles);
    auto H_t = getV<20>(Particles);
    auto Df1 = getV<21>(Particles);
    auto Df2 = getV<22>(Particles);
    auto Df3 = getV<23>(Particles);
    auto Df4 = getV<24>(Particles);
    auto Df5 = getV<25>(Particles);
    auto Df6 = getV<26>(Particles);
    auto delmu = getV<27>(Particles);
    auto k1 = getV<28>(Particles);
    auto k2 = getV<29>(Particles);
    auto k3 = getV<30>(Particles);
    auto k4 = getV<31>(Particles);
    auto H_p_b = getV<32>(Particles);
    auto FranckEnergyDensity = getV<33>(Particles);
    auto r = getV<34>(Particles);
    auto q = getV<35>(Particles);


    double eta = 1.0;
    double nu = -0.5;
    double gama = 0.1;
    double zeta = 0.07;
    double Ks = 1.0;
    double Kt = 1.1;
    double Kb = 1.5;
    double lambda = 0.1;
    //double delmu = -1.0;
    g = 0;
    delmu = -1.0;
    P = 0;
    V = 0;
    // Here fill up the boxes for particle boundary detection.
    Particles.ghost_get<ExtForce, 27>(SKIP_LABELLING);

    // Here fill up the boxes for particle detection.

    Box<3, double> up(
            {box.getLow(0) - spacing / 2.0, box.getHigh(1) - spacing / 2.0, box.getLow(2) - spacing / 2.0},
            {box.getHigh(0) + spacing / 2.0, box.getHigh(1) + spacing / 2.0, box.getHigh(2) + spacing / 2.0});

    Box<3, double> down(
            {box.getLow(0) - spacing / 2.0, box.getLow(1) - spacing / 2.0, box.getLow(2) - spacing / 2.0},
            {box.getHigh(0) + spacing / 2.0, box.getLow(1) + spacing / 2.0, box.getHigh(2) + spacing / 2.0});

    Box<3, double> left(
            {box.getLow(0) - spacing / 2.0, box.getLow(1) - spacing / 2.0, box.getLow(2) - spacing / 2.0},
            {box.getLow(0) + spacing / 2.0, box.getHigh(1) + spacing / 2.0, box.getHigh(2) + spacing / 2.0});

    Box<3, double> right(
            {box.getHigh(0) - spacing / 2.0, box.getLow(1) - spacing / 2.0, box.getLow(2) - spacing / 2.0},
            {box.getHigh(0) + spacing / 2.0, box.getHigh(1) + spacing / 2.0, box.getHigh(2) + spacing / 2.0});

    Box<3, double> front(
            {box.getLow(0) - spacing / 2.0, box.getLow(1) - spacing / 2.0, box.getLow(2) - spacing / 2.0},
            {box.getHigh(0) + spacing / 2.0, box.getHigh(1) + spacing / 2.0, box.getLow(2) + spacing / 2.0});

    Box<3, double> back(
            {box.getLow(0) - spacing / 2.0, box.getLow(1) - spacing / 2.0, box.getHigh(2) - spacing / 2.0},
            {box.getHigh(0) + spacing / 2.0, box.getHigh(1) + spacing / 2.0, box.getHigh(2) + spacing / 2.0});

    openfpm::vector<Box<3, double>> boxes;
    boxes.add(up);
    boxes.add(down);
    boxes.add(left);
    boxes.add(right);
    boxes.add(front);
    boxes.add(back);
    VTKWriter<openfpm::vector<Box<3, double>>, VECTOR_BOX> vtk_box;
    vtk_box.add(boxes);
    vtk_box.write("boxes_3d.vtk");
    auto it2 = Particles.getDomainIterator();

    while (it2.isNext()) {
        auto p = it2.get();
        Point<3, double> xp = Particles.getPos(p);
        Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * xp[x] - Lx) / Lx) - sin((2 * xp[y] - Ly) / Ly)));
        Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * xp[x] - Lx) / Lx) - sin((2 * xp[y] - Ly) / Ly)));
        Particles.getProp<0>(p)[z] = 0;
        if (front.isInside(xp) == true) {
            Boundary.add();
            Boundary.last().get<0>() = p.getKey();
        } else if (back.isInside(xp) == true) {
            Boundary.add();
            Boundary.last().get<0>() = p.getKey();
        } else if (left.isInside(xp) == true) {
            Boundary.add();
            Boundary.last().get<0>() = p.getKey();
        } else if (right.isInside(xp) == true) {
            Boundary.add();
            Boundary.last().get<0>() = p.getKey();
        } else if (up.isInside(xp) == true) {
            Boundary.add();
            Boundary.last().get<0>() = p.getKey();
        } else if (down.isInside(xp) == true) {
            Boundary.add();
            Boundary.last().get<0>() = p.getKey();
        } else {
            bulk.add();
            bulk.last().get<0>() = p.getKey();
        }
        ++it2;
    }

    vector_dist_subset<3, double, aggregate<VectorS<3, double>, VectorS<3, double>, double[3][3], VectorS<3, double>, double, double[3][3], double[3][3], VectorS<3, double>, VectorS<3, double>, VectorS<3, double>, VectorS<3, double>, double, double, double, double, double, double, double, VectorS<3, double>, double, double, double[3], double[3], double[3], double[3], double[3], double[3], double, VectorS<3, double>, VectorS<3, double>, VectorS<3, double>, VectorS<3, double>, double, double, double, double[3][3],double[3][3]>> Particles_subset(
            Particles, bulk);
    auto Pol_bulk = getV<0>(Particles_subset);
    auto P_bulk = getV<Pressure>(Particles_subset);
    auto dPol_bulk = getV<8>(Particles_subset);
    auto RHS_bulk = getV<10>(Particles_subset);

    //Particles.write_frame("Active3d_Parts", 0);

    //Particles_subset.write("Pars");
    Derivative_x Dx(Particles, ord, rCut, sampling_factor, support_options::RADIUS), Bulk_Dx(Particles_subset, ord,
                                                                                             rCut, sampling_factor,
                                                                                             support_options::RADIUS);
    Derivative_y Dy(Particles, ord, rCut, sampling_factor, support_options::RADIUS), Bulk_Dy(Particles_subset, ord,
                                                                                             rCut, sampling_factor,
                                                                                             support_options::RADIUS);
    Derivative_z Dz(Particles, ord, rCut, sampling_factor, support_options::RADIUS), Bulk_Dz(Particles_subset, ord,
                                                                                             rCut, sampling_factor,
                                                                                             support_options::RADIUS);
    Derivative_xy Dxy(Particles, ord, rCut2, sampling_factor2, support_options::RADIUS);
    Derivative_yz Dyz(Particles, ord, rCut2, sampling_factor2, support_options::RADIUS);
    Derivative_xz Dxz(Particles, ord, rCut2, sampling_factor2, support_options::RADIUS);
    auto Dyx = Dxy;
    auto Dzy = Dyz;
    auto Dzx = Dxz;

    Derivative_xx Dxx(Particles, ord, rCut2, sampling_factor2,
                      support_options::RADIUS);//, Dxx2(Particles, ord2, rCut2, sampling_factor2, support_options::RADIUS);
    Derivative_yy Dyy(Particles, ord, rCut2, sampling_factor2,
                      support_options::RADIUS);
    Derivative_zz Dzz(Particles, ord, rCut2, sampling_factor2,
                      support_options::RADIUS);

    V_t = V;
    auto px=Pol[x];
    auto py=Pol[y];
    auto pz=Pol[z];

    auto dxpx=Dx(Pol[x]);
    auto dxpy=Dx(Pol[y]);
    auto dxpz=Dx(Pol[z]);
    auto dypx=Dy(Pol[x]);
    auto dypy=Dy(Pol[y]);
    auto dypz=Dy(Pol[z]);
    auto dzpx=Dz(Pol[x]);
    auto dzpy=Dz(Pol[y]);
    auto dzpz=Dz(Pol[z]);
    auto dxxpx=Dxx(Pol[x]);
    auto dxxpy=Dxx(Pol[y]);
    auto dxxpz=Dxx(Pol[z]);
    auto dyypx=Dyy(Pol[x]);
    auto dyypy=Dyy(Pol[y]);
    auto dyypz=Dyy(Pol[z]);
    auto dzzpx=Dzz(Pol[x]);
    auto dzzpy=Dzz(Pol[y]);
    auto dzzpz=Dzz(Pol[z]);
    auto dxypx=Dxy(Pol[x]);
    auto dxypy=Dxy(Pol[y]);
    auto dxypz=Dxy(Pol[z]);
    auto dxzpx=Dxz(Pol[x]);
    auto dxzpy=Dxz(Pol[y]);
    auto dxzpz=Dxz(Pol[z]);
    auto dyzpx=Dyz(Pol[x]);
    auto dyzpy=Dyz(Pol[y]);
    auto dyzpz=Dyz(Pol[z]);
    auto dxhx=Dx(h[x]);
    auto dxhy=Dx(h[y]);
    auto dxhz=Dx(h[z]);
    auto dyhx=Dy(h[x]);
    auto dyhy=Dy(h[y]);
    auto dyhz=Dy(h[z]);
    auto dzhx=Dz(h[x]);
    auto dzhy=Dz(h[y]);
    auto dzhz=Dz(h[z]);

    auto dxqxx=Dx(Pol[x]*Pol[x]-1/3*(Pol[x]*Pol[x]+Pol[y]*Pol[y]+Pol[z]*Pol[z]));
    auto dyqxy=Dy(Pol[x]*Pol[x]);
    auto dzqxz=Dz(Pol[x]*Pol[z]);
    auto dxqyx=Dx(Pol[y]*Pol[x]);
    auto dyqyy=Dy(Pol[y]*Pol[y]-1/3*(Pol[x]*Pol[x]+Pol[y]*Pol[y]+Pol[z]*Pol[z]));
    auto dzqyz=Dz(Pol[y]*Pol[z]);
    auto dxqzx=Dx(Pol[z]*Pol[x]);
    auto dyqzy=Dy(Pol[z]*Pol[y]);
    auto dzqzz=Dz(Pol[z]*Pol[z]-1/3*(Pol[x]*Pol[x]+Pol[y]*Pol[y]+Pol[z]*Pol[z]));


    eq_id vx, vy, vz;

    vx.setId(0);
    vy.setId(1);
    vz.setId(2);
    timer tt;
    double V_err_eps = 5e-2;
    double V_err = 1, V_err_old;
    int n = 0;
    int nmax = 300;
    int ctr = 0, errctr, Vreset = 0;
    double tim = 0;
    double tf = 1.024;
    div = 0;
    double sum, sum1, sum_k;
    while (tim <= tf) {
        tt.start();
        petsc_solver<double> solverPetsc;
        solverPetsc.setSolver(KSPGMRES);
        //solverPetsc.setRestart(250);
        solverPetsc.setPreconditioner(PCJACOBI);
        Particles.ghost_get<Polarization>(SKIP_LABELLING);



        FranckEnergyDensity = 0.5*Ks*(dxpx + dypy + dzpz)*(dxpx + dypy + dzpz) +
                              0.5*Kt*((dypz - dzpy)*px + (-dxpz + dzpx)*py + (dxpy - dypx)*pz)*((dypz - dzpy)*px + (-dxpz + dzpx)*py + (dxpy - dypx)*pz) +
                              0.5*Kb*((-dxpz*px + dzpx*px - dypz*py + dzpy*py)*(-dxpz*px + dzpx*px - dypz*py + dzpy*py) +
                                      (dxpy*py - dypx*py + dxpz*pz - dzpx*pz)*(dxpy*py - dypx*py + dxpz*pz - dzpx*pz) +
                                      (-dxpy*px + dypx*px + dypz*pz - dzpy*pz)*(-dxpy*px + dypx*px + dypz*pz - dzpy*pz));

        h[x]=Ks*(dxxpx + dxypy + dxzpz) +
             Kb*((-dxypy - dxzpz + dyypx + dzzpx)*px*px + (-dxypy + dyypx)*py*py + (dypy*dzpx + dxpy*(dypz - 2*dzpy) + dypx*dzpy + dxpz*(-dypy - 2*dzpz) + 2*dzpx*dzpz)*pz + (-dxzpz + dzzpx)*pz*pz +
                 py*(dypz*dzpx + dxpz*(-2*dypz + dzpy) + dxpy*(-2*dypy - dzpz) + dypx*(2*dypy + dzpz) + (-dxypz - dxzpy + 2*dyzpx)*pz) +
                 px*(-dxpy*dxpy - dxpz*dxpz + dypx*dypx + dypz*dypz + dzpx*dzpx - 2*dypz*dzpy + dzpy*dzpy + (-dyzpz + dzzpy)*py + (dyypz - dyzpy)*pz)) +
             Kt*((-dxzpz + dzzpx)*py*py + (dxpz*dypy -  dypy*dzpx + dypx*(2*dypz -  dzpy) + dxpy*(-3*dypz + 2*dzpy))*pz + (- dxypy +  dyypx)*pz*pz + py*(-dypz*dzpx + dxpz*(2*dypz - 3*dzpy) + 2*dzpx*dzpy +  dxpy*dzpz -   dypx*dzpz + ( dxypz +  dxzpy - 2*dyzpx)*pz) +
                 px*(-2*dypz*dypz + 4*dypz*dzpy - 2*dzpy*dzpy + ( dyzpz -  dzzpy)*py + (- dyypz + dyzpy)*pz));

        h[y]= Ks*(dxypx + dyypy + dyzpz) +
              Kb*((dxxpy - dxypx)*px*px + (dxxpy - dxypx - dyzpz + dzzpy)*py*py + (dxpz*dypx + dxpy*dzpx - 2*dypx*dzpx + dxpx*(-dypz + dzpy) - 2*dypz*dzpz + 2*dzpy*dzpz)*pz + (-dyzpz + dzzpy)*pz*pz +
                  py*(dxpy*dxpy + dxpz*dxpz - dypx*dypx - dypz*dypz - 2*dxpz*dzpx + dzpx*dzpx + dzpy*dzpy + (dxxpz - dxzpx)*pz) +
                  px*(dxpx*(2*dxpy - 2*dypx) + dypz*dzpx + dxpz*(-2*dypz + dzpy) + dxpy*dzpz - dypx*dzpz + (-dxzpz + dzzpx)*py + (-dxypz + 2*dxzpy - dyzpx)*pz)) +
              Kt*((-dyzpz + dzzpy)*px*px + (-3*dxpz*dypx + dxpy*(2*dxpz - dzpx) + 2*dypx*dzpx + dxpx*(dypz - dzpy))*pz + (dxxpy - dxypx)*pz*pz + py*(-2*dxpz*dxpz + 4*dxpz*dzpx - 2*dzpx*dzpx + (-dxxpz + dxzpx)*pz) +
                  px*(-3*dypz*dzpx + dxpz*(2*dypz - dzpy) + 2*dzpx*dzpy - dxpy*dzpz + dypx*dzpz + (dxzpz - dzzpx)*py + (dxypz - 2*dxzpy + dyzpx)*pz));


        h[z]=Ks*(dxzpx + dyzpy + dzzpz) +
             Kb*((dxxpz - dxzpx)*px*px + (dyypz - dyzpy)*py*py + (dxpy*dxpy + dxpz*dxpz - 2*dxpy*dypx + dypx*dypx + dypz*dypz - dzpx*dzpx - dzpy*dzpy)*pz + (dxxpz - dxzpx + dyypz - dyzpy)*pz*pz +
                 py*(dxpz*dypx + dxpy*dzpx - 2*dypx*dzpx + dypy*(2*dypz - 2*dzpy) + dxpx*(dypz - dzpy) + (dxxpy - dxypx)*pz) +
                 px*(dxpz*dypy + dxpx*(2*dxpz - 2*dzpx) - dypy*dzpx + dxpy*(dypz - 2*dzpy) + dypx*dzpy + (2*dxypz - dxzpy - dyzpx)*py + (-dxypy + dyypx)*pz))+
             Kt*((dyypz - dyzpy)*px*px + (dxxpz - dxzpx)*py*py + (-2*dxpy*dxpy + 4*dxpy*dypx - 2*dypx*dypx)*pz + py*(-dxpz*dypx + dxpy*(2*dxpz - 3*dzpx) + 2*dypx*dzpx + dxpx*(-dypz + dzpy) + (-dxxpy + dxypx)*pz) +
                 px*(-dxpz*dypy + dypy*dzpx + dypx*(2*dypz - 3*dzpy) + dxpy*(-dypz + 2*dzpy) + (-2*dxypz + dxzpy + dyzpx)*py + (dxypy - dyypx)*pz));

        //Particles.ghost_get<33>(SKIP_LABELLING);
        sigma[x][x] =
                -dxpx*(dxpx + dypy + dzpz)*Ks -
                dxpy*Kt*pz*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) -
                dxpz*Kt*py*(-dypz*px + dzpy*px + dxpz*py - dzpx*py - dxpy*pz + dypx*pz) -
                0.5*dxpz*Kb*(2*px*(dxpz*px - dzpx*px + (dypz - dzpy)*py) + 2*pz*(dxpy*py - dypx*py + (dxpz - dzpx)*pz)) -
                0.5*dxpy*Kb*(2*py*(dxpy*py - dypx*py + (dxpz - dzpx)*pz) + 2*px*(dxpy*px - dypx*px + (-dypz + dzpy)*pz));
        sigma[x][y] =
                -dxpy*(dxpx + dypy + dzpz)*Ks - dxpz*Kb*py*(dxpz*px - dzpx*px + (dypz - dzpy)*py) +
                dxpx*Kt*pz*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) +
                dxpz*Kt*px*(-dypz*px + dzpy*px + dxpz*py - dzpx*py - dxpy*pz + dypx*pz) +
                dxpx*Kb*py*(dxpy*py - dypx*py + (dxpz - dzpx)*pz) -
                dxpz*Kb*pz*(-dxpy*px + dypx*px + (dypz - dzpy)*pz) +
                dxpx*Kb*px*(dxpy*px - dypx*px + (-dypz + dzpy)*pz);

        sigma[x][z] =
                -dxpz*(dxpx + dypy + dzpz)*Ks +
                dxpy*Kt*px*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) +
                dxpx*Kt*py*(-dypz*px + dzpy*px + dxpz*py - dzpx*py - dxpy*pz + dypx*pz) -
                0.5*dxpx*Kb*(2*px*(-dxpz*px + dzpx*px + (-dypz + dzpy)*py) - 2*pz*(dxpy*py - dypx*py + (dxpz - dzpx)*pz)) -
                0.5*dxpy*Kb*(2*py*(-dxpz*px + dzpx*px + (-dypz + dzpy)*py) - 2*pz*(-dxpy*px + dypx*px + (dypz - dzpy)*pz));
        sigma[y][x] =
                -dypx*(dxpx + dypy + dzpz)*Ks +
                dypz*Kt*py*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) -
                dypy*Kt*pz*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) -
                0.5*dypz*Kb*(2*px*(dxpz*px - dzpx*px + (dypz - dzpy)*py) + 2*pz*(dxpy*py - dypx*py + (dxpz - dzpx)*pz)) -
                0.5*dypy*Kb*(2*py*(dxpy*py - dypx*py + (dxpz - dzpx)*pz) + 2*px*(dxpy*px - dypx*px + (-dypz + dzpy)*pz));
        sigma[y][y] =
                -dypy*(dxpx + dypy + dzpz)*Ks -
                dypz*Kb*py*(dxpz*px - dzpx*px + (dypz - dzpy)*py) -
                dypz*Kt*px*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) -
                dypx*Kt*pz*(-dypz*px + dzpy*px + dxpz*py - dzpx*py -dxpy*pz + dypx*pz) -
                dypx*Kb*py*(-dxpy*py + dypx*py + (-dxpz + dzpx)*pz) -
                dypx*Kb*px*(-dxpy*px + dypx*px + (dypz - dzpy)*pz) -
                dypz*Kb*pz*(-dxpy*px + dypx*px + (dypz - dzpy)*pz);
        sigma[y][z] =
                -dypz*(dxpx + dypy + dzpz)*Ks +
                dypy*Kt*px*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) +
                dypx*Kt*py*(-dypz*px + dzpy*px + dxpz*py - dzpx*py -dxpy*pz + dypx*pz) -
                0.5*dypx*Kb*(2*px*(-dxpz*px + dzpx*px + (-dypz + dzpy)*py) - 2*pz*(dxpy*py - dypx*py + (dxpz - dzpx)*pz)) -
                0.5*dypy*Kb*(2*py*(-dxpz*px + dzpx*px + (-dypz + dzpy)*py) - 2*pz*(-dxpy*px + dypx*px + (dypz - dzpy)*pz));
        sigma[z][x] =
                -dzpx*(dxpx + dypy + dzpz)*Ks -
                dzpz*Kt*py*(-dypz*px + dzpy*px + dxpz*py - dzpx*py - dxpy*pz + dypx*pz) +
                dzpy*Kt*pz*(-dypz*px + dzpy*px + dxpz*py - dzpx*py - dxpy*pz + dypx*pz) -
                0.5*dzpz*Kb*(2*px*(dxpz*px - dzpx*px + (dypz - dzpy)*py) + 2*pz*(dxpy*py - dypx*py + (dxpz - dzpx)*pz)) -
                0.5*dzpy*Kb*(2*py*(dxpy*py - dypx*py + (dxpz - dzpx)*pz) + 2*px*(dxpy*px - dypx*px + (-dypz + dzpy)*pz));
        sigma[z][y] =
                -dzpy*(dxpx + dypy + dzpz)*Ks -
                dzpz*Kb*py*(dxpz*px - dzpx*px + (dypz - dzpy)*py) -
                dzpz*Kt*px*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) +
                dzpx*Kt*pz*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) -
                dzpx*Kb*py*(-dxpy*py + dypx*py + (-dxpz + dzpx)*pz) -
                dzpz*Kb*pz*(-dxpy*px + dypx*px + (dypz - dzpy)*pz) +
                dzpx*Kb*px*(dxpy*px - dypx*px + (-dypz + dzpy)*pz);

        sigma[z][z] =
                -dzpz*(dxpx + dypy + dzpz)*Ks -
                dzpx*Kt*py*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) -
                dzpy*Kt*px*(-dypz*px + dzpy*px + dxpz*py - dzpx*py - dxpy*pz + dypx*pz) -
                0.5*dzpx*Kb*(2*px*(-dxpz*px + dzpx*px + (-dypz + dzpy)*py) - 2*pz*(dxpy*py - dypx*py + (dxpz - dzpx)*pz)) -
                0.5*dzpy*Kb*(2*py*(-dxpz*px + dzpx*px + (-dypz + dzpy)*py) - 2*pz*(-dxpy*px + dypx*px + (dypz - dzpy)*pz));


        Particles.ghost_get<Stress>(SKIP_LAB            tt.start();
        petsc_solver<double> solverPetsc;
        solverPetsc.setSolver(KSPGMRES);
        //solverPetsc.setRestart(250);
        solverPetsc.setPreconditioner(PCJACOBI);
        Particles.ghost_get<Polarization>(SKIP_LABELLING);



        FranckEnergyDensity = 0.5*Ks*(dxpx + dypy + dzpz)*(dxpx + dypy + dzpz) +
                              0.5*Kt*((dypz - dzpy)*px + (-dxpz + dzpx)*py + (dxpy - dypx)*pz)*((dypz - dzpy)*px + (-dxpz + dzpx)*py + (dxpy - dypx)*pz) +
                              0.5*Kb*((-dxpz*px + dzpx*px - dypz*py + dzpy*py)*(-dxpz*px + dzpx*px - dypz*py + dzpy*py) +
                                      (dxpy*py - dypx*py + dxpz*pz - dzpx*pz)*(dxpy*py - dypx*py + dxpz*pz - dzpx*pz) +
                                      (-dxpy*px + dypx*px + dypz*pz - dzpy*pz)*(-dxpy*px + dypx*px + dypz*pz - dzpy*pz));

        h[x]=Ks*(dxxpx + dxypy + dxzpz) +
             Kb*((-dxypy - dxzpz + dyypx + dzzpx)*px*px + (-dxypy + dyypx)*py*py + (dypy*dzpx + dxpy*(dypz - 2*dzpy) + dypx*dzpy + dxpz*(-dypy - 2*dzpz) + 2*dzpx*dzpz)*pz + (-dxzpz + dzzpx)*pz*pz +
                 py*(dypz*dzpx + dxpz*(-2*dypz + dzpy) + dxpy*(-2*dypy - dzpz) + dypx*(2*dypy + dzpz) + (-dxypz - dxzpy + 2*dyzpx)*pz) +
                 px*(-dxpy*dxpy - dxpz*dxpz + dypx*dypx + dypz*dypz + dzpx*dzpx - 2*dypz*dzpy + dzpy*dzpy + (-dyzpz + dzzpy)*py + (dyypz - dyzpy)*pz)) +
             Kt*((-dxzpz + dzzpx)*py*py + (dxpz*dypy -  dypy*dzpx + dypx*(2*dypz -  dzpy) + dxpy*(-3*dypz + 2*dzpy))*pz + (- dxypy +  dyypx)*pz*pz + py*(-dypz*dzpx + dxpz*(2*dypz - 3*dzpy) + 2*dzpx*dzpy +  dxpy*dzpz -   dypx*dzpz + ( dxypz +  dxzpy - 2*dyzpx)*pz) +
                 px*(-2*dypz*dypz + 4*dypz*dzpy - 2*dzpy*dzpy + ( dyzpz -  dzzpy)*py + (- dyypz + dyzpy)*pz));

        h[y]= Ks*(dxypx + dyypy + dyzpz) +
              Kb*((dxxpy - dxypx)*px*px + (dxxpy - dxypx - dyzpz + dzzpy)*py*py + (dxpz*dypx + dxpy*dzpx - 2*dypx*dzpx + dxpx*(-dypz + dzpy) - 2*dypz*dzpz + 2*dzpy*dzpz)*pz + (-dyzpz + dzzpy)*pz*pz +
                  py*(dxpy*dxpy + dxpz*dxpz - dypx*dypx - dypz*dypz - 2*dxpz*dzpx + dzpx*dzpx + dzpy*dzpy + (dxxpz - dxzpx)*pz) +
                  px*(dxpx*(2*dxpy - 2*dypx) + dypz*dzpx + dxpz*(-2*dypz + dzpy) + dxpy*dzpz - dypx*dzpz + (-dxzpz + dzzpx)*py + (-dxypz + 2*dxzpy - dyzpx)*pz)) +
              Kt*((-dyzpz + dzzpy)*px*px + (-3*dxpz*dypx + dxpy*(2*dxpz - dzpx) + 2*dypx*dzpx + dxpx*(dypz - dzpy))*pz + (dxxpy - dxypx)*pz*pz + py*(-2*dxpz*dxpz + 4*dxpz*dzpx - 2*dzpx*dzpx + (-dxxpz + dxzpx)*pz) +
                  px*(-3*dypz*dzpx + dxpz*(2*dypz - dzpy) + 2*dzpx*dzpy - dxpy*dzpz + dypx*dzpz + (dxzpz - dzzpx)*py + (dxypz - 2*dxzpy + dyzpx)*pz));


        h[z]=Ks*(dxzpx + dyzpy + dzzpz) +
             Kb*((dxxpz - dxzpx)*px*px + (dyypz - dyzpy)*py*py + (dxpy*dxpy + dxpz*dxpz - 2*dxpy*dypx + dypx*dypx + dypz*dypz - dzpx*dzpx - dzpy*dzpy)*pz + (dxxpz - dxzpx + dyypz - dyzpy)*pz*pz +
                 py*(dxpz*dypx + dxpy*dzpx - 2*dypx*dzpx + dypy*(2*dypz - 2*dzpy) + dxpx*(dypz - dzpy) + (dxxpy - dxypx)*pz) +
                 px*(dxpz*dypy + dxpx*(2*dxpz - 2*dzpx) - dypy*dzpx + dxpy*(dypz - 2*dzpy) + dypx*dzpy + (2*dxypz - dxzpy - dyzpx)*py + (-dxypy + dyypx)*pz))+
             Kt*((dyypz - dyzpy)*px*px + (dxxpz - dxzpx)*py*py + (-2*dxpy*dxpy + 4*dxpy*dypx - 2*dypx*dypx)*pz + py*(-dxpz*dypx + dxpy*(2*dxpz - 3*dzpx) + 2*dypx*dzpx + dxpx*(-dypz + dzpy) + (-dxxpy + dxypx)*pz) +
                 px*(-dxpz*dypy + dypy*dzpx + dypx*(2*dypz - 3*dzpy) + dxpy*(-dypz + 2*dzpy) + (-2*dxypz + dxzpy + dyzpx)*py + (dxypy - dyypx)*pz));

        //Particles.ghost_get<33>(SKIP_LABELLING);
        sigma[x][x] =
                -dxpx*(dxpx + dypy + dzpz)*Ks -
                dxpy*Kt*pz*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) -
                dxpz*Kt*py*(-dypz*px + dzpy*px + dxpz*py - dzpx*py - dxpy*pz + dypx*pz) -
                0.5*dxpz*Kb*(2*px*(dxpz*px - dzpx*px + (dypz - dzpy)*py) + 2*pz*(dxpy*py - dypx*py + (dxpz - dzpx)*pz)) -
                0.5*dxpy*Kb*(2*py*(dxpy*py - dypx*py + (dxpz - dzpx)*pz) + 2*px*(dxpy*px - dypx*px + (-dypz + dzpy)*pz));
        sigma[x][y] =
                -dxpy*(dxpx + dypy + dzpz)*Ks - dxpz*Kb*py*(dxpz*px - dzpx*px + (dypz - dzpy)*py) +
                dxpx*Kt*pz*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) +
                dxpz*Kt*px*(-dypz*px + dzpy*px + dxpz*py - dzpx*py - dxpy*pz + dypx*pz) +
                dxpx*Kb*py*(dxpy*py - dypx*py + (dxpz - dzpx)*pz) -
                dxpz*Kb*pz*(-dxpy*px + dypx*px + (dypz - dzpy)*pz) +
                dxpx*Kb*px*(dxpy*px - dypx*px + (-dypz + dzpy)*pz);

        sigma[x][z] =
                -dxpz*(dxpx + dypy + dzpz)*Ks +
                dxpy*Kt*px*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) +
                dxpx*Kt*py*(-dypz*px + dzpy*px + dxpz*py - dzpx*py - dxpy*pz + dypx*pz) -
                0.5*dxpx*Kb*(2*px*(-dxpz*px + dzpx*px + (-dypz + dzpy)*py) - 2*pz*(dxpy*py - dypx*py + (dxpz - dzpx)*pz)) -
                0.5*dxpy*Kb*(2*py*(-dxpz*px + dzpx*px + (-dypz + dzpy)*py) - 2*pz*(-dxpy*px + dypx*px + (dypz - dzpy)*pz));
        sigma[y][x] =
                -dypx*(dxpx + dypy + dzpz)*Ks +
                dypz*Kt*py*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) -
                dypy*Kt*pz*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) -
                0.5*dypz*Kb*(2*px*(dxpz*px - dzpx*px + (dypz - dzpy)*py) + 2*pz*(dxpy*py - dypx*py + (dxpz - dzpx)*pz)) -
                0.5*dypy*Kb*(2*py*(dxpy*py - dypx*py + (dxpz - dzpx)*pz) + 2*px*(dxpy*px - dypx*px + (-dypz + dzpy)*pz));
        sigma[y][y] =
                -dypy*(dxpx + dypy + dzpz)*Ks -
                dypz*Kb*py*(dxpz*px - dzpx*px + (dypz - dzpy)*py) -
                dypz*Kt*px*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) -
                dypx*Kt*pz*(-dypz*px + dzpy*px + dxpz*py - dzpx*py -dxpy*pz + dypx*pz) -
                dypx*Kb*py*(-dxpy*py + dypx*py + (-dxpz + dzpx)*pz) -
                dypx*Kb*px*(-dxpy*px + dypx*px + (dypz - dzpy)*pz) -
                dypz*Kb*pz*(-dxpy*px + dypx*px + (dypz - dzpy)*pz);
        sigma[y][z] =
                -dypz*(dxpx + dypy + dzpz)*Ks +
                dypy*Kt*px*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) +
                dypx*Kt*py*(-dypz*px + dzpy*px + dxpz*py - dzpx*py -dxpy*pz + dypx*pz) -
                0.5*dypx*Kb*(2*px*(-dxpz*px + dzpx*px + (-dypz + dzpy)*py) - 2*pz*(dxpy*py - dypx*py + (dxpz - dzpx)*pz)) -
                0.5*dypy*Kb*(2*py*(-dxpz*px + dzpx*px + (-dypz + dzpy)*py) - 2*pz*(-dxpy*px + dypx*px + (dypz - dzpy)*pz));
        sigma[z][x] =
                -dzpx*(dxpx + dypy + dzpz)*Ks -
                dzpz*Kt*py*(-dypz*px + dzpy*px + dxpz*py - dzpx*py - dxpy*pz + dypx*pz) +
                dzpy*Kt*pz*(-dypz*px + dzpy*px + dxpz*py - dzpx*py - dxpy*pz + dypx*pz) -
                0.5*dzpz*Kb*(2*px*(dxpz*px - dzpx*px + (dypz - dzpy)*py) + 2*pz*(dxpy*py - dypx*py + (dxpz - dzpx)*pz)) -
                0.5*dzpy*Kb*(2*py*(dxpy*py - dypx*py + (dxpz - dzpx)*pz) + 2*px*(dxpy*px - dypx*px + (-dypz + dzpy)*pz));
        sigma[z][y] =
                -dzpy*(dxpx + dypy + dzpz)*Ks -
                dzpz*Kb*py*(dxpz*px - dzpx*px + (dypz - dzpy)*py) -
                dzpz*Kt*px*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) +
                dzpx*Kt*pz*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) -
                dzpx*Kb*py*(-dxpy*py + dypx*py + (-dxpz + dzpx)*pz) -
                dzpz*Kb*pz*(-dxpy*px + dypx*px + (dypz - dzpy)*pz) +
                dzpx*Kb*px*(dxpy*px - dypx*px + (-dypz + dzpy)*pz);

        sigma[z][z] =
                -dzpz*(dxpx + dypy + dzpz)*Ks -
                dzpx*Kt*py*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) -
                dzpy*Kt*px*(-dypz*px + dzpy*px + dxpz*py - dzpx*py - dxpy*pz + dypx*pz) -
                0.5*dzpx*Kb*(2*px*(-dxpz*px + dzpx*px + (-dypz + dzpy)*py) - 2*pz*(dxpy*py - dypx*py + (dxpz - dzpx)*pz)) -
                0.5*dzpy*Kb*(2*py*(-dxpz*px + dzpx*px + (-dypz + dzpy)*py) - 2*pz*(-dxpy*px + dypx*px + (dypz - dzpy)*pz));


        Particles.ghost_get<Stress>(SKIP_LABELLING);
        Particles.ghost_get<MolField>(SKIP_LABELLING);

        dV[x] = 0.5*(-dypy*h[x] + dypx*h[y] + dyhy*px - dyhx*py) + 0.5*(-dzpz*h[x] + dzpx*h[z] + dzhz*px - dzhx*pz) +
                zeta*delmu*(dxqxx + dyqxy + dzqxz)  +
                nu*(1/2*(-dypy*h[x] + dypx*h[y] + dyhy*px - dyhx*py) + 1/3*(-dxpx*h[x] - dxpy*h[y] - dxpz*h[z] - dxhx*px - dxhy*py -dxhz*pz) + 1/2*(-dzpz*h[x] + dzpx*h[z] + dzhz*px - dzhx*pz))+
                Dx(sigma[x][x])+Dy(sigma[x][y])+Dz(sigma[x][z]);


        dV[y] = 0.5*(dxpy*h[x] - dxpx*h[y] - dxhy*px + dxhx*py) + 0.5*(-dzpz*h[y] + dzpy*h[z] + dzhz*py - dzhy*pz) +
                zeta*delmu*(dxqyx + dyqyy + dzqyz) +
                nu*(1/2*(dxpy*h[x] - dxpx*h[y] - dxhy*px + dxhx*py) + 1/3*(-dypx*h[x] - dypy*h[y] - dypz*h[z] - dyhx*px - dyhy*py - dyhz*pz) + 1/2*(-dzpz*h[y] + dzpy*h[z] + dzhz*py - dzhy*pz))+
                Dx(sigma[y][x])+Dy(sigma[y][y])+Dz(sigma[y][z]);

        dV[z]=0.5*(dxpz*h[x] - dxpx*h[z] - dxhz*px + dxhx*pz) + 0.5*(dypz*h[y] - dypy*h[z] - dyhz*py + dyhy*pz) +
              zeta*delmu*(dxqzx + dyqzy +dzqzz)  +
              nu*(1/2*(dxpz*h[x] - dxpx*h[z] - dxhz*px + dxhx*pz)+ 1/2*(dypz*h[y] - dypy*h[z] - dyhz*py + dyhy*pz) + 1/3*(-dzpx*h[x] - dzpy*h[y] - dzpz*h[z] - dzhx*px - dzhy*py -dzhz*pz))+
              Dx(sigma[z][x])+Dy(sigma[z][y])+Dz(sigma[z][z]);

        Particles.ghost_get<9>(SKIP_LABELLING);


        //Particles.write("PolarI");
        //Velocity Solution n iterations



        auto Stokes1 = eta * (2*Dxx(V[x]) + Dxy(V[y]) + Dyy(V[x]) + Dxz(V[z]) + Dzz(V[x]));
        auto Stokes2 = eta * (2*Dyy(V[y]) + Dxx(V[y]) + Dxy(V[x]) + Dyz(V[z]) + Dzz(V[y]));
        auto Stokes3 = eta * (2*Dzz(V[z]) + Dxx(V[z]) + Dxz(V[x]) + Dyy(V[z]) + Dyz(V[y]));
        std::cout << "Init of Velocity took " << tt.getwct() << " seconds." << std::endl;
        tt.start();
        V_err = 1;
        n = 0;
        errctr = 0;
        if (Vreset == 1) {
            P_bulk = 0;
            P = 0;
            Vreset = 0;
        }
        P = 0;
        P_bulk = 0;

        while (V_err >= V_err_eps && n <= nmax) {
            RHS_bulk[x] = -dV[x]+Bulk_Dx(P);
            RHS_bulk[y] = -dV[y]+Bulk_Dy(P);
            RHS_bulk[z] = -dV[z]+Bulk_Dz(P);
            Particles.ghost_get<10>(SKIP_LABELLING);
            DCPSE_scheme<equations3d3, decltype(Particles)> Solver(Particles);
            Solver.impose(Stokes1, bulk, RHS[0], vx);
            Solver.impose(Stokes2, bulk, RHS[1], vy);
            Solver.impose(Stokes3, bulk, RHS[2], vz);
            Solver.impose(V[x], Boundary, 0, vx);
            Solver.impose(V[y], Boundary, 0, vy);
            Solver.impose(V[z], Boundary, 0, vx);
            Solver.solve_with_solver(solverPetsc, V[x], V[y], V[z]);
            Particles.ghost_get<Velocity>(SKIP_LABELLING);
            div = -(Dx(V[x]) + Dy(V[y])+Dz(V[z]));
            P_bulk = P + div;
            sum = 0;
            sum1 = 0;
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                sum += (Particles.getProp<18>(p)[0] - Particles.getProp<1>(p)[0]) *
                       (Particles.getProp<18>(p)[0] - Particles.getProp<1>(p)[0]) +
                       (Particles.getProp<18>(p)[1] - Particles.getProp<1>(p)[1]) *
                       (Particles.getProp<18>(p)[1] - Particles.getProp<1>(p)[1]) +
                       (Particles.getProp<18>(p)[2] - Particles.getProp<1>(p)[2]) *
                       (Particles.getProp<18>(p)[2] - Particles.getProp<1>(p)[2])

                        ;
                sum1 += Particles.getProp<1>(p)[0] * Particles.getProp<1>(p)[0] +
                        Particles.getProp<1>(p)[1] * Particles.getProp<1>(p)[1]+
                        Particles.getProp<1>(p)[2] * Particles.getProp<1>(p)[2];
            }
            sum = sqrt(sum);
            sum1 = sqrt(sum1);

            v_cl.sum(sum);
            v_cl.sum(sum1);
            v_cl.execute();
            V_t = V;
            Particles.ghost_get<1, 4, 18>(SKIP_LABELLING);
            V_err_old = V_err;
            V_err = sum / sum1;
            if (V_err > V_err_old || abs(V_err_old - V_err) < 1e-8) {
                errctr++;
                //alpha_P -= 0.1;
            } else {
                errctr = 0;
            }
            if (n > 3) {
                if (errctr > 3) {
                    std::cout << "CONVERGENCE LOOP BROKEN DUE TO INCREASE/VERY SLOW DECREASE IN ERROR" << std::endl;
                    Vreset = 1;
                    break;
                } else {
                    Vreset = 0;
                }
            }
            n++;
            //Particles.write_frame("V_debug", n);
            if (v_cl.rank() == 0) {
                std::cout << "Rel l2 cgs err in V = " << V_err << " at " << n << std::endl;
            }
        }
        tt.stop();
        u[x][x] = Dx(V[x]);
        u[x][y] = 0.5 * (Dx(V[y]) + Dy(V[x]));
        u[x][z] = 0.5 * (Dx(V[z]) + Dz(V[x]));
        u[y][x] = 0.5 * (Dy(V[x]) + Dx(V[y]));
        u[y][y] = Dy(V[y]);
        u[y][z] = 0.5 * (Dy(V[z]) + Dz(V[y]));
        u[z][x] = 0.5 * (Dz(V[x]) + Dx(V[z]));
        u[z][y] = 0.5 * (Dz(V[y]) + Dy(V[z]));
        u[z][z] = Dz(V[z]);

        if (v_cl.rank() == 0) {
            std::cout << "Rel l2 cgs err in V = " << V_err << " and took " << tt.getwct() << " seconds with " << n
                      << " iterations. dt is set to " << dt
                      << std::endl;
        }

        W[x][x] = 0;
        W[x][y] = 0.5 * (Dy(V[x]) - Dx(V[y]));
        W[x][z] = 0.5 * (Dz(V[x]) - Dx(V[z]));
        W[y][x] = 0.5 * (Dx(V[y]) - Dy(V[x]));
        W[y][y] = 0;
        W[y][z] = 0.5 * (Dz(V[y]) - Dy(V[z]));
        W[z][x] = 0.5 * (Dx(V[z]) - Dz(V[x]));
        W[z][y] = 0.5 * (Dy(V[z]) - Dz(V[y]));
        W[z][z] = 0;

        Particles.deleteGhost();
        Particles.write_frame("Polar3d", ctr);
        Particles.ghost_get<0>();
        ctr++;
        //auto lambda = -1/(9*gama)*(-3*h[x]*Pol[x]-3*h[y]*Pol[y]-3*h[z]*Pol[z]+
        //        gama*nu*(Pol[x]*Pol[x]*u[x][x]+Pol[y]*Pol[y]*u[y][y]+Pol[z]*Pol[z]*u[z][z]+
        //        Pol[x]*(Pol[y]*(u[x][y]+u[y][x])+Pol[z]*(u[x][z]+u[z][x])) +Pol[y]*Pol[z]*(u[y][z]+u[z][y])))/(Pol[x]*Pol[x]+Pol[y]*Pol[y]+Pol[z]*Pol[z]);
        dPol=Pol;
        Particles.ghost_get<8>(SKIP_LABELLING);
        k1[x] = h[x]/gama-nu*(Pol[x]*u[x][x]+Pol[y]*u[x][y]+Pol[z]*u[x][z]) + lambda*Pol[x]/delmu + (W[x][x]*Pol[x]+W[x][y]*Pol[y]+W[x][z]*Pol[z]);
        k1[y] = h[y]/gama-nu*(Pol[x]*u[y][x]+Pol[y]*u[y][y]+Pol[z]*u[y][z]) + lambda*Pol[y]/delmu + (W[y][x]*Pol[x]+W[y][y]*Pol[y]+W[y][z]*Pol[z]);
        k1[z] = h[z]/gama-nu*(Pol[x]*u[z][x]+Pol[y]*u[z][y]+Pol[z]*u[z][z]) + lambda*Pol[z]/delmu + (W[z][x]*Pol[x]+W[z][y]*Pol[y]+W[z][z]*Pol[z]);
        Pol_bulk = dPol + (0.5 * dt) * k1;
        Particles.ghost_get<0>(SKIP_LABELLING);

        h[x]=Ks*(dxxpx + dxypy + dxzpz) +
             Kb*((-dxypy - dxzpz + dyypx + dzzpx)*px*px + (-dxypy + dyypx)*py*py + (dypy*dzpx + dxpy*(dypz - 2*dzpy) + dypx*dzpy + dxpz*(-dypy - 2*dzpz) + 2*dzpx*dzpz)*pz + (-dxzpz + dzzpx)*pz*pz +
                 py*(dypz*dzpx + dxpz*(-2*dypz + dzpy) + dxpy*(-2*dypy - dzpz) + dypx*(2*dypy + dzpz) + (-dxypz - dxzpy + 2*dyzpx)*pz) +
                 px*(-dxpy*dxpy - dxpz*dxpz + dypx*dypx + dypz*dypz + dzpx*dzpx - 2*dypz*dzpy + dzpy*dzpy + (-dyzpz + dzzpy)*py + (dyypz - dyzpy)*pz)) +
             Kt*((-dxzpz + dzzpx)*py*py + (dxpz*dypy -  dypy*dzpx + dypx*(2*dypz -  dzpy) + dxpy*(-3*dypz + 2*dzpy))*pz + (- dxypy +  dyypx)*pz*pz + py*(-dypz*dzpx + dxpz*(2*dypz - 3*dzpy) + 2*dzpx*dzpy +  dxpy*dzpz -   dypx*dzpz + ( dxypz +  dxzpy - 2*dyzpx)*pz) +
                 px*(-2*dypz*dypz + 4*dypz*dzpy - 2*dzpy*dzpy + ( dyzpz -  dzzpy)*py + (- dyypz + dyzpy)*pz));

        h[y]= Ks*(dxypx + dyypy + dyzpz) +
              Kb*((dxxpy - dxypx)*px*px + (dxxpy - dxypx - dyzpz + dzzpy)*py*py + (dxpz*dypx + dxpy*dzpx - 2*dypx*dzpx + dxpx*(-dypz + dzpy) - 2*dypz*dzpz + 2*dzpy*dzpz)*pz + (-dyzpz + dzzpy)*pz*pz +
                  py*(dxpy*dxpy + dxpz*dxpz - dypx*dypx - dypz*dypz - 2*dxpz*dzpx + dzpx*dzpx + dzpy*dzpy + (dxxpz - dxzpx)*pz) +
                  px*(dxpx*(2*dxpy - 2*dypx) + dypz*dzpx + dxpz*(-2*dypz + dzpy) + dxpy*dzpz - dypx*dzpz + (-dxzpz + dzzpx)*py + (-dxypz + 2*dxzpy - dyzpx)*pz)) +
              Kt*((-dyzpz + dzzpy)*px*px + (-3*dxpz*dypx + dxpy*(2*dxpz - dzpx) + 2*dypx*dzpx + dxpx*(dypz - dzpy))*pz + (dxxpy - dxypx)*pz*pz + py*(-2*dxpz*dxpz + 4*dxpz*dzpx - 2*dzpx*dzpx + (-dxxpz + dxzpx)*pz) +
                  px*(-3*dypz*dzpx + dxpz*(2*dypz - dzpy) + 2*dzpx*dzpy - dxpy*dzpz + dypx*dzpz + (dxzpz - dzzpx)*py + (dxypz - 2*dxzpy + dyzpx)*pz));


        h[z]=Ks*(dxzpx + dyzpy + dzzpz) +
             Kb*((dxxpz - dxzpx)*px*px + (dyypz - dyzpy)*py*py + (dxpy*dxpy + dxpz*dxpz - 2*dxpy*dypx + dypx*dypx + dypz*dypz - dzpx*dzpx - dzpy*dzpy)*pz + (dxxpz - dxzpx + dyypz - dyzpy)*pz*pz +
                 py*(dxpz*dypx + dxpy*dzpx - 2*dypx*dzpx + dypy*(2*dypz - 2*dzpy) + dxpx*(dypz - dzpy) + (dxxpy - dxypx)*pz) +
                 px*(dxpz*dypy + dxpx*(2*dxpz - 2*dzpx) - dypy*dzpx + dxpy*(dypz - 2*dzpy) + dypx*dzpy + (2*dxypz - dxzpy - dyzpx)*py + (-dxypy + dyypx)*pz))+
             Kt*((dyypz - dyzpy)*px*px + (dxxpz - dxzpx)*py*py + (-2*dxpy*dxpy + 4*dxpy*dypx - 2*dypx*dypx)*pz + py*(-dxpz*dypx + dxpy*(2*dxpz - 3*dzpx) + 2*dypx*dzpx + dxpx*(-dypz + dzpy) + (-dxxpy + dxypx)*pz) +
                 px*(-dxpz*dypy + dypy*dzpx + dypx*(2*dypz - 3*dzpy) + dxpy*(-dypz + 2*dzpy) + (-2*dxypz + dxzpy + dyzpx)*py + (dxypy - dyypx)*pz));


        k2[x] = h[x]/gama-nu*(Pol[x]*u[x][x]+Pol[y]*u[x][y]+Pol[z]*u[x][z]) + lambda*Pol[x]/delmu + (W[x][x]*Pol[x]+W[x][y]*Pol[y]+W[x][z]*Pol[z]);
        k2[y] = h[y]/gama-nu*(Pol[x]*u[y][x]+Pol[y]*u[y][y]+Pol[z]*u[y][z]) + lambda*Pol[y]/delmu + (W[y][x]*Pol[x]+W[y][y]*Pol[y]+W[y][z]*Pol[z]);
        k2[z] = h[z]/gama-nu*(Pol[x]*u[z][x]+Pol[y]*u[z][y]+Pol[z]*u[z][z]) + lambda*Pol[z]/delmu + (W[z][x]*Pol[x]+W[z][y]*Pol[y]+W[z][z]*Pol[z]);
        Pol_bulk = dPol + (0.5 * dt) * k2;
        Particles.ghost_get<0>(SKIP_LABELLING);
        h[x]=Ks*(dxxpx + dxypy + dxzpz) +
             Kb*((-dxypy - dxzpz + dyypx + dzzpx)*px*px + (-dxypy + dyypx)*py*py + (dypy*dzpx + dxpy*(dypz - 2*dzpy) + dypx*dzpy + dxpz*(-dypy - 2*dzpz) + 2*dzpx*dzpz)*pz + (-dxzpz + dzzpx)*pz*pz +
                 py*(dypz*dzpx + dxpz*(-2*dypz + dzpy) + dxpy*(-2*dypy - dzpz) + dypx*(2*dypy + dzpz) + (-dxypz - dxzpy + 2*dyzpx)*pz) +
                 px*(-dxpy*dxpy - dxpz*dxpz + dypx*dypx + dypz*dypz + dzpx*dzpx - 2*dypz*dzpy + dzpy*dzpy + (-dyzpz + dzzpy)*py + (dyypz - dyzpy)*pz)) +
             Kt*((-dxzpz + dzzpx)*py*py + (dxpz*dypy -  dypy*dzpx + dypx*(2*dypz -  dzpy) + dxpy*(-3*dypz + 2*dzpy))*pz + (- dxypy +  dyypx)*pz*pz + py*(-dypz*dzpx + dxpz*(2*dypz - 3*dzpy) + 2*dzpx*dzpy +  dxpy*dzpz -   dypx*dzpz + ( dxypz +  dxzpy - 2*dyzpx)*pz) +
                 px*(-2*dypz*dypz + 4*dypz*dzpy - 2*dzpy*dzpy + ( dyzpz -  dzzpy)*py + (- dyypz + dyzpy)*pz));

        h[y]= Ks*(dxypx + dyypy + dyzpz) +
              Kb*((dxxpy - dxypx)*px*px + (dxxpy - dxypx - dyzpz + dzzpy)*py*py + (dxpz*dypx + dxpy*dzpx - 2*dypx*dzpx + dxpx*(-dypz + dzpy) - 2*dypz*dzpz + 2*dzpy*dzpz)*pz + (-dyzpz + dzzpy)*pz*pz +
                  py*(dxpy*dxpy + dxpz*dxpz - dypx*dypx - dypz*dypz - 2*dxpz*dzpx + dzpx*dzpx + dzpy*dzpy + (dxxpz - dxzpx)*pz) +
                  px*(dxpx*(2*dxpy - 2*dypx) + dypz*dzpx + dxpz*(-2*dypz + dzpy) + dxpy*dzpz - dypx*dzpz + (-dxzpz + dzzpx)*py + (-dxypz + 2*dxzpy - dyzpx)*pz)) +
              Kt*((-dyzpz + dzzpy)*px*px + (-3*dxpz*dypx + dxpy*(2*dxpz - dzpx) + 2*dypx*dzpx + dxpx*(dypz - dzpy))*pz + (dxxpy - dxypx)*pz*pz + py*(-2*dxpz*dxpz + 4*dxpz*dzpx - 2*dzpx*dzpx + (-dxxpz + dxzpx)*pz) +
                  px*(-3*dypz*dzpx + dxpz*(2*dypz - dzpy) + 2*dzpx*dzpy - dxpy*dzpz + dypx*dzpz + (dxzpz - dzzpx)*py + (dxypz - 2*dxzpy + dyzpx)*pz));


        h[z]=Ks*(dxzpx + dyzpy + dzzpz) +
             Kb*((dxxpz - dxzpx)*px*px + (dyypz - dyzpy)*py*py + (dxpy*dxpy + dxpz*dxpz - 2*dxpy*dypx + dypx*dypx + dypz*dypz - dzpx*dzpx - dzpy*dzpy)*pz + (dxxpz - dxzpx + dyypz - dyzpy)*pz*pz +
                 py*(dxpz*dypx + dxpy*dzpx - 2*dypx*dzpx + dypy*(2*dypz - 2*dzpy) + dxpx*(dypz - dzpy) + (dxxpy - dxypx)*pz) +
                 px*(dxpz*dypy + dxpx*(2*dxpz - 2*dzpx) - dypy*dzpx + dxpy*(dypz - 2*dzpy) + dypx*dzpy + (2*dxypz - dxzpy - dyzpx)*py + (-dxypy + dyypx)*pz))+
             Kt*((dyypz - dyzpy)*px*px + (dxxpz - dxzpx)*py*py + (-2*dxpy*dxpy + 4*dxpy*dypx - 2*dypx*dypx)*pz + py*(-dxpz*dypx + dxpy*(2*dxpz - 3*dzpx) + 2*dypx*dzpx + dxpx*(-dypz + dzpy) + (-dxxpy + dxypx)*pz) +
                 px*(-dxpz*dypy + dypy*dzpx + dypx*(2*dypz - 3*dzpy) + dxpy*(-dypz + 2*dzpy) + (-2*dxypz + dxzpy + dyzpx)*py + (dxypy - dyypx)*pz));


        k3[x] = h[x]/gama-nu*(Pol[x]*u[x][x]+Pol[y]*u[x][y]+Pol[z]*u[x][z]) + lambda*Pol[x]/delmu + (W[x][x]*Pol[x]+W[x][y]*Pol[y]+W[x][z]*Pol[z]);
        k3[y] = h[y]/gama-nu*(Pol[x]*u[y][x]+Pol[y]*u[y][y]+Pol[z]*u[y][z]) + lambda*Pol[y]/delmu + (W[y][x]*Pol[x]+W[y][y]*Pol[y]+W[y][z]*Pol[z]);
        k3[z] = h[z]/gama-nu*(Pol[x]*u[z][x]+Pol[y]*u[z][y]+Pol[z]*u[z][z]) + lambda*Pol[z]/delmu + (W[z][x]*Pol[x]+W[z][y]*Pol[y]+W[z][z]*Pol[z]);
        Pol_bulk = dPol + (dt)*k3;

        Particles.ghost_get<0>(SKIP_LABELLING);
        h[x]=Ks*(dxxpx + dxypy + dxzpz) +
             Kb*((-dxypy - dxzpz + dyypx + dzzpx)*px*px + (-dxypy + dyypx)*py*py + (dypy*dzpx + dxpy*(dypz - 2*dzpy) + dypx*dzpy + dxpz*(-dypy - 2*dzpz) + 2*dzpx*dzpz)*pz + (-dxzpz + dzzpx)*pz*pz +
                 py*(dypz*dzpx + dxpz*(-2*dypz + dzpy) + dxpy*(-2*dypy - dzpz) + dypx*(2*dypy + dzpz) + (-dxypz - dxzpy + 2*dyzpx)*pz) +
                 px*(-dxpy*dxpy - dxpz*dxpz + dypx*dypx + dypz*dypz + dzpx*dzpx - 2*dypz*dzpy + dzpy*dzpy + (-dyzpz + dzzpy)*py + (dyypz - dyzpy)*pz)) +
             Kt*((-dxzpz + dzzpx)*py*py + (dxpz*dypy -  dypy*dzpx + dypx*(2*dypz -  dzpy) + dxpy*(-3*dypz + 2*dzpy))*pz + (- dxypy +  dyypx)*pz*pz + py*(-dypz*dzpx + dxpz*(2*dypz - 3*dzpy) + 2*dzpx*dzpy +  dxpy*dzpz -   dypx*dzpz + ( dxypz +  dxzpy - 2*dyzpx)*pz) +
                 px*(-2*dypz*dypz + 4*dypz*dzpy - 2*dzpy*dzpy + ( dyzpz -  dzzpy)*py + (- dyypz + dyzpy)*pz));

        h[y]= Ks*(dxypx + dyypy + dyzpz) +
              Kb*((dxxpy - dxypx)*px*px + (dxxpy - dxypx - dyzpz + dzzpy)*py*py + (dxpz*dypx + dxpy*dzpx - 2*dypx*dzpx + dxpx*(-dypz + dzpy) - 2*dypz*dzpz + 2*dzpy*dzpz)*pz + (-dyzpz + dzzpy)*pz*pz +
                  py*(dxpy*dxpy + dxpz*dxpz - dypx*dypx - dypz*dypz - 2*dxpz*dzpx + dzpx*dzpx + dzpy*dzpy + (dxxpz - dxzpx)*pz) +
                  px*(dxpx*(2*dxpy - 2*dypx) + dypz*dzpx + dxpz*(-2*dypz + dzpy) + dxpy*dzpz - dypx*dzpz + (-dxzpz + dzzpx)*py + (-dxypz + 2*dxzpy - dyzpx)*pz)) +
              Kt*((-dyzpz + dzzpy)*px*px + (-3*dxpz*dypx + dxpy*(2*dxpz - dzpx) + 2*dypx*dzpx + dxpx*(dypz - dzpy))*pz + (dxxpy - dxypx)*pz*pz + py*(-2*dxpz*dxpz + 4*dxpz*dzpx - 2*dzpx*dzpx + (-dxxpz + dxzpx)*pz) +
                  px*(-3*dypz*dzpx + dxpz*(2*dypz - dzpy) + 2*dzpx*dzpy - dxpy*dzpz + dypx*dzpz + (dxzpz - dzzpx)*py + (dxypz - 2*dxzpy + dyzpx)*pz));


        h[z]=Ks*(dxzpx + dyzpy + dzzpz) +
             Kb*((dxxpz - dxzpx)*px*px + (dyypz - dyzpy)*py*py + (dxpy*dxpy + dxpz*dxpz - 2*dxpy*dypx + dypx*dypx + dypz*dypz - dzpx*dzpx - dzpy*dzpy)*pz + (dxxpz - dxzpx + dyypz - dyzpy)*pz*pz +
                 py*(dxpz*dypx + dxpy*dzpx - 2*dypx*dzpx + dypy*(2*dypz - 2*dzpy) + dxpx*(dypz - dzpy) + (dxxpy - dxypx)*pz) +
                 px*(dxpz*dypy + dxpx*(2*dxpz - 2*dzpx) - dypy*dzpx + dxpy*(dypz - 2*dzpy) + dypx*dzpy + (2*dxypz - dxzpy - dyzpx)*py + (-dxypy + dyypx)*pz))+
             Kt*((dyypz - dyzpy)*px*px + (dxxpz - dxzpx)*py*py + (-2*dxpy*dxpy + 4*dxpy*dypx - 2*dypx*dypx)*pz + py*(-dxpz*dypx + dxpy*(2*dxpz - 3*dzpx) + 2*dypx*dzpx + dxpx*(-dypz + dzpy) + (-dxxpy + dxypx)*pz) +
                 px*(-dxpz*dypy + dypy*dzpx + dypx*(2*dypz - 3*dzpy) + dxpy*(-dypz + 2*dzpy) + (-2*dxypz + dxzpy + dyzpx)*py + (dxypy - dyypx)*pz));

        k4[x] = h[x]/gama-nu*(Pol[x]*u[x][x]+Pol[y]*u[x][y]+Pol[z]*u[x][z]) + lambda*Pol[x]/delmu + (W[x][x]*Pol[x]+W[x][y]*Pol[y]+W[x][z]*Pol[z]);
        k4[y] = h[y]/gama-nu*(Pol[x]*u[y][x]+Pol[y]*u[y][y]+Pol[z]*u[y][z]) + lambda*Pol[y]/delmu + (W[y][x]*Pol[x]+W[y][y]*Pol[y]+W[y][z]*Pol[z]);
        k4[z] = h[z]/gama-nu*(Pol[x]*u[z][x]+Pol[y]*u[z][y]+Pol[z]*u[z][z]) + lambda*Pol[z]/delmu + (W[z][x]*Pol[x]+W[z][y]*Pol[y]+W[z][z]*Pol[z]);

        Pol = dPol + (dt / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4);
        for (int j = 0; j < Boundary.size(); j++) {
            auto p = Boundary.get<0>(j);
            Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                         sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                         sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            Particles.getProp<0>(p)[z] =0;
        }

        k1 = V;
        k2 = 0.5 * dt * k1 + V;
        k3 = 0.5 * dt * k2 + V;
        k4 = dt * k3 + V;
        Pos = Pos + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);

        Particles.map();

        Particles.ghost_get<0, ExtForce, 27>();
        indexUpdate(Particles,Boundary, bulk, up, down, left,right,front,back);
        vector_dist_subset<3, double, aggregate<VectorS<3, double>, VectorS<3, double>, double[3][3], VectorS<3, double>, double, double[3][3], double[3][3], VectorS<3, double>, VectorS<3, double>, VectorS<3, double>, VectorS<3, double>, double, double, double, double, double, double, double, VectorS<3, double>, double, double, double[3], double[3], double[3], double[3], double[3], double[3], double, VectorS<3, double>, VectorS<3, double>, VectorS<3, double>, VectorS<3, double>, double, double, double, double[3][3],double[3][3]>> Particles_subset(
                Particles, bulk);
        auto Pol_bulk = getV<0>(Particles_subset);
        auto P_bulk = getV<Pressure>(Particles_subset);
        auto dPol_bulk = getV<8>(Particles_subset);
        auto RHS_bulk = getV<10>(Particles_subset);

        //Particles_subset.write("debug");

        tt.start();
        Dx.update(Particles);
        Dy.update(Particles);
        Dz.update(Particles);
        Dxy.update(Particles);
        Dxz.update(Particles);
        Dyz.update(Particles);
        auto Dyx = Dxy;
        auto Dzy = Dyz;
        auto Dzx = Dxz;
        Dxx.update(Particles);
        Dyy.update(Particles);
        Dzz.update(Particles);

        Bulk_Dx.update(Particles_subset);
        Bulk_Dy.update(Particles_subset);
        Bulk_Dz.update(Particles_subset);


        tt.stop();
        if (v_cl.rank() == 0) {
            std::cout << "Updation of operators took " << tt.getwct() << " seconds." << std::endl;
            std::cout << "Time step " << ctr - 1 << " : " << tim << " over." << std::endl;
            std::cout << "----------------------------------------------------------" << std::endl;
        }

        tim += dt;
        ELLING);
        Particles.ghost_get<MolField>(SKIP_LABELLING);

        dV[x] = 0.5*(-dypy*h[x] + dypx*h[y] + dyhy*px - dyhx*py) + 0.5*(-dzpz*h[x] + dzpx*h[z] + dzhz*px - dzhx*pz) +
                zeta*delmu*(dxqxx + dyqxy + dzqxz)  +
                nu*(1/2*(-dypy*h[x] + dypx*h[y] + dyhy*px - dyhx*py) + 1/3*(-dxpx*h[x] - dxpy*h[y] - dxpz*h[z] - dxhx*px - dxhy*py -dxhz*pz) + 1/2*(-dzpz*h[x] + dzpx*h[z] + dzhz*px - dzhx*pz))+
                Dx(sigma[x][x])+Dy(sigma[x][y])+Dz(sigma[x][z]);


        dV[y] = 0.5*(dxpy*h[x] - dxpx*h[y] - dxhy*px + dxhx*py) + 0.5*(-dzpz*h[y] + dzpy*h[z] + dzhz*py - dzhy*pz) +
                zeta*delmu*(dxqyx + dyqyy + dzqyz) +
                nu*(1/2*(dxpy*h[x] - dxpx*h[y] - dxhy*px + dxhx*py) + 1/3*(-dypx*h[x] - dypy*h[y] - dypz*h[z] - dyhx*px - dyhy*py - dyhz*pz) + 1/2*(-dzpz*h[y] + dzpy*h[z] + dzhz*py - dzhy*pz))+
                Dx(sigma[y][x])+Dy(sigma[y][y])+Dz(sigma[y][z]);

        dV[z]=0.5*(dxpz*h[x] - dxpx*h[z] - dxhz*px + dxhx*pz) + 0.5*(dypz*h[y] - dypy*h[z] - dyhz*py + dyhy*pz) +
              zeta*delmu*(dxqzx + dyqzy +dzqzz)  +
              nu*(1/2*(dxpz*h[x] - dxpx*h[z] - dxhz*px + dxhx*pz)+ 1/2*(dypz*h[y] - dypy*h[z] - dyhz*py + dyhy*pz) + 1/3*(-dzpx*h[x] - dzpy*h[y] - dzpz*h[z] - dzhx*px - dzhy*py -dzhz*pz))+
              Dx(sigma[z][x])+Dy(sigma[z][y])+Dz(sigma[z][z]);

        Particles.ghost_get<9>(SKIP_LABELLING);


        //Particles.write("PolarI");
        //Velocity Solution n iterations



        auto Stokes1 = eta * (2*Dxx(V[x]) + Dxy(V[y]) + Dyy(V[x]) + Dxz(V[z]) + Dzz(V[x]));
        auto Stokes2 = eta * (2*Dyy(V[y]) + Dxx(V[y]) + Dxy(V[x]) + Dyz(V[z]) + Dzz(V[y]));
        auto Stokes3 = eta * (2*Dzz(V[z]) + Dxx(V[z]) + Dxz(V[x]) + Dyy(V[z]) + Dyz(V[y]));
        std::cout << "Init of Velocity took " << tt.getwct() << " seconds." << std::endl;
        tt.start();
        V_err = 1;
        n = 0;
        errctr = 0;
        if (Vreset == 1) {
            P_bulk = 0;
            P = 0;
            Vreset = 0;
        }
        P = 0;
        P_bulk = 0;

        while (V_err >= V_err_eps && n <= nmax) {
            RHS_bulk[x] = -dV[x]+Bulk_Dx(P);
            RHS_bulk[y] = -dV[y]+Bulk_Dy(P);
            RHS_bulk[z] = -dV[z]+Bulk_Dz(P);
            Particles.ghost_get<10>(SKIP_LABELLING);
            DCPSE_scheme<equations3d3, decltype(Particles)> Solver(Particles);
            Solver.impose(Stokes1, bulk, RHS[0], vx);
            Solver.impose(Stokes2, bulk, RHS[1], vy);
            Solver.impose(Stokes3, bulk, RHS[2], vz);
            Solver.impose(V[x], Boundary, 0, vx);
            Solver.impose(V[y], Boundary, 0, vy);
            Solver.impose(V[z], Boundary, 0, vx);
            Solver.solve_with_solver(solverPetsc, V[x], V[y], V[z]);
            Particles.ghost_get<Velocity>(SKIP_LABELLING);
            div = -(Dx(V[x]) + Dy(V[y])+Dz(V[z]));
            P_bulk = P + div;
            sum = 0;
            sum1 = 0;
            for (int j = 0; j < bulk.size(); j++) {
                auto p = bulk.get<0>(j);
                sum += (Particles.getProp<18>(p)[0] - Particles.getProp<1>(p)[0]) *
                       (Particles.getProp<18>(p)[0] - Particles.getProp<1>(p)[0]) +
                       (Particles.getProp<18>(p)[1] - Particles.getProp<1>(p)[1]) *
                       (Particles.getProp<18>(p)[1] - Particles.getProp<1>(p)[1]) +
                       (Particles.getProp<18>(p)[2] - Particles.getProp<1>(p)[2]) *
                       (Particles.getProp<18>(p)[2] - Particles.getProp<1>(p)[2])

                        ;
                sum1 += Particles.getProp<1>(p)[0] * Particles.getProp<1>(p)[0] +
                        Particles.getProp<1>(p)[1] * Particles.getProp<1>(p)[1]+
                        Particles.getProp<1>(p)[2] * Particles.getProp<1>(p)[2];
            }
            sum = sqrt(sum);
            sum1 = sqrt(sum1);

            v_cl.sum(sum);
            v_cl.sum(sum1);
            v_cl.execute();
            V_t = V;
            Particles.ghost_get<1, 4, 18>(SKIP_LABELLING);
            V_err_old = V_err;
            V_err = sum / sum1;
            if (V_err > V_err_old || abs(V_err_old - V_err) < 1e-8) {
                errctr++;
                //alpha_P -= 0.1;
            } else {
                errctr = 0;
            }
            if (n > 3) {
                if (errctr > 3) {
                    std::cout << "CONVERGENCE LOOP BROKEN DUE TO INCREASE/VERY SLOW DECREASE IN ERROR" << std::endl;
                    Vreset = 1;
                    break;
                } else {
                    Vreset = 0;
                }
            }
            n++;
            //Particles.write_frame("V_debug", n);
            if (v_cl.rank() == 0) {
                std::cout << "Rel l2 cgs err in V = " << V_err << " at " << n << std::endl;
            }
        }
        tt.stop();
        u[x][x] = Dx(V[x]);
        u[x][y] = 0.5 * (Dx(V[y]) + Dy(V[x]));
        u[x][z] = 0.5 * (Dx(V[z]) + Dz(V[x]));
        u[y][x] = 0.5 * (Dy(V[x]) + Dx(V[y]));
        u[y][y] = Dy(V[y]);
        u[y][z] = 0.5 * (Dy(V[z]) + Dz(V[y]));
        u[z][x] = 0.5 * (Dz(V[x]) + Dx(V[z]));
        u[z][y] = 0.5 * (Dz(V[y]) + Dy(V[z]));
        u[z][z] = Dz(V[z]);

        if (v_cl.rank() == 0) {
            std::cout << "Rel l2 cgs err in V = " << V_err << " and took " << tt.getwct() << " seconds with " << n
                      << " iterations. dt is set to " << dt
                      << std::endl;
        }

        W[x][x] = 0;
        W[x][y] = 0.5 * (Dy(V[x]) - Dx(V[y]));
        W[x][z] = 0.5 * (Dz(V[x]) - Dx(V[z]));
        W[y][x] = 0.5 * (Dx(V[y]) - Dy(V[x]));
        W[y][y] = 0;
        W[y][z] = 0.5 * (Dz(V[y]) - Dy(V[z]));
        W[z][x] = 0.5 * (Dx(V[z]) - Dz(V[x]));
        W[z][y] = 0.5 * (Dy(V[z]) - Dz(V[y]));
        W[z][z] = 0;

        Particles.deleteGhost();
        Particles.write_frame("Polar3d", ctr);
        Particles.ghost_get<0>();
        ctr++;
        //auto lambda = -1/(9*gama)*(-3*h[x]*Pol[x]-3*h[y]*Pol[y]-3*h[z]*Pol[z]+
        //        gama*nu*(Pol[x]*Pol[x]*u[x][x]+Pol[y]*Pol[y]*u[y][y]+Pol[z]*Pol[z]*u[z][z]+
        //        Pol[x]*(Pol[y]*(u[x][y]+u[y][x])+Pol[z]*(u[x][z]+u[z][x])) +Pol[y]*Pol[z]*(u[y][z]+u[z][y])))/(Pol[x]*Pol[x]+Pol[y]*Pol[y]+Pol[z]*Pol[z]);
        dPol=Pol;
        Particles.ghost_get<8>(SKIP_LABELLING);
        k1[x] = h[x]/gama-nu*(Pol[x]*u[x][x]+Pol[y]*u[x][y]+Pol[z]*u[x][z]) + lambda*Pol[x]/delmu + (W[x][x]*Pol[x]+W[x][y]*Pol[y]+W[x][z]*Pol[z]);
        k1[y] = h[y]/gama-nu*(Pol[x]*u[y][x]+Pol[y]*u[y][y]+Pol[z]*u[y][z]) + lambda*Pol[y]/delmu + (W[y][x]*Pol[x]+W[y][y]*Pol[y]+W[y][z]*Pol[z]);
        k1[z] = h[z]/gama-nu*(Pol[x]*u[z][x]+Pol[y]*u[z][y]+Pol[z]*u[z][z]) + lambda*Pol[z]/delmu + (W[z][x]*Pol[x]+W[z][y]*Pol[y]+W[z][z]*Pol[z]);
        Pol_bulk = dPol + (0.5 * dt) * k1;
        Particles.ghost_get<0>(SKIP_LABELLING);

        h[x]=Ks*(dxxpx + dxypy + dxzpz) +
             Kb*((-dxypy - dxzpz + dyypx + dzzpx)*px*px + (-dxypy + dyypx)*py*py + (dypy*dzpx + dxpy*(dypz - 2*dzpy) + dypx*dzpy + dxpz*(-dypy - 2*dzpz) + 2*dzpx*dzpz)*pz + (-dxzpz + dzzpx)*pz*pz +
                 py*(dypz*dzpx + dxpz*(-2*dypz + dzpy) + dxpy*(-2*dypy - dzpz) + dypx*(2*dypy + dzpz) + (-dxypz - dxzpy + 2*dyzpx)*pz) +
                 px*(-dxpy*dxpy - dxpz*dxpz + dypx*dypx + dypz*dypz + dzpx*dzpx - 2*dypz*dzpy + dzpy*dzpy + (-dyzpz + dzzpy)*py + (dyypz - dyzpy)*pz)) +
             Kt*((-dxzpz + dzzpx)*py*py + (dxpz*dypy -  dypy*dzpx + dypx*(2*dypz -  dzpy) + dxpy*(-3*dypz + 2*dzpy))*pz + (- dxypy +  dyypx)*pz*pz + py*(-dypz*dzpx + dxpz*(2*dypz - 3*dzpy) + 2*dzpx*dzpy +  dxpy*dzpz -   dypx*dzpz + ( dxypz +  dxzpy - 2*dyzpx)*pz) +
                 px*(-2*dypz*dypz + 4*dypz*dzpy - 2*dzpy*dzpy + ( dyzpz -  dzzpy)*py + (- dyypz + dyzpy)*pz));

        h[y]= Ks*(dxypx + dyypy + dyzpz) +
              Kb*((dxxpy - dxypx)*px*px + (dxxpy - dxypx - dyzpz + dzzpy)*py*py + (dxpz*dypx + dxpy*dzpx - 2*dypx*dzpx + dxpx*(-dypz + dzpy) - 2*dypz*dzpz + 2*dzpy*dzpz)*pz + (-dyzpz + dzzpy)*pz*pz +
                  py*(dxpy*dxpy + dxpz*dxpz - dypx*dypx - dypz*dypz - 2*dxpz*dzpx + dzpx*dzpx + dzpy*dzpy + (dxxpz - dxzpx)*pz) +
                  px*(dxpx*(2*dxpy - 2*dypx) + dypz*dzpx + dxpz*(-2*dypz + dzpy) + dxpy*dzpz - dypx*dzpz + (-dxzpz + dzzpx)*py + (-dxypz + 2*dxzpy - dyzpx)*pz)) +
              Kt*((-dyzpz + dzzpy)*px*px + (-3*dxpz*dypx + dxpy*(2*dxpz - dzpx) + 2*dypx*dzpx + dxpx*(dypz - dzpy))*pz + (dxxpy - dxypx)*pz*pz + py*(-2*dxpz*dxpz + 4*dxpz*dzpx - 2*dzpx*dzpx + (-dxxpz + dxzpx)*pz) +
                  px*(-3*dypz*dzpx + dxpz*(2*dypz - dzpy) + 2*dzpx*dzpy - dxpy*dzpz + dypx*dzpz + (dxzpz - dzzpx)*py + (dxypz - 2*dxzpy + dyzpx)*pz));


        h[z]=Ks*(dxzpx + dyzpy + dzzpz) +
             Kb*((dxxpz - dxzpx)*px*px + (dyypz - dyzpy)*py*py + (dxpy*dxpy + dxpz*dxpz - 2*dxpy*dypx + dypx*dypx + dypz*dypz - dzpx*dzpx - dzpy*dzpy)*pz + (dxxpz - dxzpx + dyypz - dyzpy)*pz*pz +
                 py*(dxpz*dypx + dxpy*dzpx - 2*dypx*dzpx + dypy*(2*dypz - 2*dzpy) + dxpx*(dypz - dzpy) + (dxxpy - dxypx)*pz) +
                 px*(dxpz*dypy + dxpx*(2*dxpz - 2*dzpx) - dypy*dzpx + dxpy*(dypz - 2*dzpy) + dypx*dzpy + (2*dxypz - dxzpy - dyzpx)*py + (-dxypy + dyypx)*pz))+
             Kt*((dyypz - dyzpy)*px*px + (dxxpz - dxzpx)*py*py + (-2*dxpy*dxpy + 4*dxpy*dypx - 2*dypx*dypx)*pz + py*(-dxpz*dypx + dxpy*(2*dxpz - 3*dzpx) + 2*dypx*dzpx + dxpx*(-dypz + dzpy) + (-dxxpy + dxypx)*pz) +
                 px*(-dxpz*dypy + dypy*dzpx + dypx*(2*dypz - 3*dzpy) + dxpy*(-dypz + 2*dzpy) + (-2*dxypz + dxzpy + dyzpx)*py + (dxypy - dyypx)*pz));


        k2[x] = h[x]/gama-nu*(Pol[x]*u[x][x]+Pol[y]*u[x][y]+Pol[z]*u[x][z]) + lambda*Pol[x]/delmu + (W[x][x]*Pol[x]+W[x][y]*Pol[y]+W[x][z]*Pol[z]);
        k2[y] = h[y]/gama-nu*(Pol[x]*u[y][x]+Pol[y]*u[y][y]+Pol[z]*u[y][z]) + lambda*Pol[y]/delmu + (W[y][x]*Pol[x]+W[y][y]*Pol[y]+W[y][z]*Pol[z]);
        k2[z] = h[z]/gama-nu*(Pol[x]*u[z][x]+Pol[y]*u[z][y]+Pol[z]*u[z][z]) + lambda*Pol[z]/delmu + (W[z][x]*Pol[x]+W[z][y]*Pol[y]+W[z][z]*Pol[z]);
        Pol_bulk = dPol + (0.5 * dt) * k2;
        Particles.ghost_get<0>(SKIP_LABELLING);
        h[x]=Ks*(dxxpx + dxypy + dxzpz) +
             Kb*((-dxypy - dxzpz + dyypx + dzzpx)*px*px + (-dxypy + dyypx)*py*py + (dypy*dzpx + dxpy*(dypz - 2*dzpy) + dypx*dzpy + dxpz*(-dypy - 2*dzpz) + 2*dzpx*dzpz)*pz + (-dxzpz + dzzpx)*pz*pz +
                 py*(dypz*dzpx + dxpz*(-2*dypz + dzpy) + dxpy*(-2*dypy - dzpz) + dypx*(2*dypy + dzpz) + (-dxypz - dxzpy + 2*dyzpx)*pz) +
                 px*(-dxpy*dxpy - dxpz*dxpz + dypx*dypx + dypz*dypz + dzpx*dzpx - 2*dypz*dzpy + dzpy*dzpy + (-dyzpz + dzzpy)*py + (dyypz - dyzpy)*pz)) +
             Kt*((-dxzpz + dzzpx)*py*py + (dxpz*dypy -  dypy*dzpx + dypx*(2*dypz -  dzpy) + dxpy*(-3*dypz + 2*dzpy))*pz + (- dxypy +  dyypx)*pz*pz + py*(-dypz*dzpx + dxpz*(2*dypz - 3*dzpy) + 2*dzpx*dzpy +  dxpy*dzpz -   dypx*dzpz + ( dxypz +  dxzpy - 2*dyzpx)*pz) +
                 px*(-2*dypz*dypz + 4*dypz*dzpy - 2*dzpy*dzpy + ( dyzpz -  dzzpy)*py + (- dyypz + dyzpy)*pz));

        h[y]= Ks*(dxypx + dyypy + dyzpz) +
              Kb*((dxxpy - dxypx)*px*px + (dxxpy - dxypx - dyzpz + dzzpy)*py*py + (dxpz*dypx + dxpy*dzpx - 2*dypx*dzpx + dxpx*(-dypz + dzpy) - 2*dypz*dzpz + 2*dzpy*dzpz)*pz + (-dyzpz + dzzpy)*pz*pz +
                  py*(dxpy*dxpy + dxpz*dxpz - dypx*dypx - dypz*dypz - 2*dxpz*dzpx + dzpx*dzpx + dzpy*dzpy + (dxxpz - dxzpx)*pz) +
                  px*(dxpx*(2*dxpy - 2*dypx) + dypz*dzpx + dxpz*(-2*dypz + dzpy) + dxpy*dzpz - dypx*dzpz + (-dxzpz + dzzpx)*py + (-dxypz + 2*dxzpy - dyzpx)*pz)) +
              Kt*((-dyzpz + dzzpy)*px*px + (-3*dxpz*dypx + dxpy*(2*dxpz - dzpx) + 2*dypx*dzpx + dxpx*(dypz - dzpy))*pz + (dxxpy - dxypx)*pz*pz + py*(-2*dxpz*dxpz + 4*dxpz*dzpx - 2*dzpx*dzpx + (-dxxpz + dxzpx)*pz) +
                  px*(-3*dypz*dzpx + dxpz*(2*dypz - dzpy) + 2*dzpx*dzpy - dxpy*dzpz + dypx*dzpz + (dxzpz - dzzpx)*py + (dxypz - 2*dxzpy + dyzpx)*pz));


        h[z]=Ks*(dxzpx + dyzpy + dzzpz) +
             Kb*((dxxpz - dxzpx)*px*px + (dyypz - dyzpy)*py*py + (dxpy*dxpy + dxpz*dxpz - 2*dxpy*dypx + dypx*dypx + dypz*dypz - dzpx*dzpx - dzpy*dzpy)*pz + (dxxpz - dxzpx + dyypz - dyzpy)*pz*pz +
                 py*(dxpz*dypx + dxpy*dzpx - 2*dypx*dzpx + dypy*(2*dypz - 2*dzpy) + dxpx*(dypz - dzpy) + (dxxpy - dxypx)*pz) +
                 px*(dxpz*dypy + dxpx*(2*dxpz - 2*dzpx) - dypy*dzpx + dxpy*(dypz - 2*dzpy) + dypx*dzpy + (2*dxypz - dxzpy - dyzpx)*py + (-dxypy + dyypx)*pz))+
             Kt*((dyypz - dyzpy)*px*px + (dxxpz - dxzpx)*py*py + (-2*dxpy*dxpy + 4*dxpy*dypx - 2*dypx*dypx)*pz + py*(-dxpz*dypx + dxpy*(2*dxpz - 3*dzpx) + 2*dypx*dzpx + dxpx*(-dypz + dzpy) + (-dxxpy + dxypx)*pz) +
                 px*(-dxpz*dypy + dypy*dzpx + dypx*(2*dypz - 3*dzpy) + dxpy*(-dypz + 2*dzpy) + (-2*dxypz + dxzpy + dyzpx)*py + (dxypy - dyypx)*pz));


        k3[x] = h[x]/gama-nu*(Pol[x]*u[x][x]+Pol[y]*u[x][y]+Pol[z]*u[x][z]) + lambda*Pol[x]/delmu + (W[x][x]*Pol[x]+W[x][y]*Pol[y]+W[x][z]*Pol[z]);
        k3[y] = h[y]/gama-nu*(Pol[x]*u[y][x]+Pol[y]*u[y][y]+Pol[z]*u[y][z]) + lambda*Pol[y]/delmu + (W[y][x]*Pol[x]+W[y][y]*Pol[y]+W[y][z]*Pol[z]);
        k3[z] = h[z]/gama-nu*(Pol[x]*u[z][x]+Pol[y]*u[z][y]+Pol[z]*u[z][z]) + lambda*Pol[z]/delmu + (W[z][x]*Pol[x]+W[z][y]*Pol[y]+W[z][z]*Pol[z]);
        Pol_bulk = dPol + (dt)*k3;

        Particles.ghost_get<0>(SKIP_LABELLING);
        h[x]=Ks*(dxxpx + dxypy + dxzpz) +
             Kb*((-dxypy - dxzpz + dyypx + dzzpx)*px*px + (-dxypy + dyypx)*py*py + (dypy*dzpx + dxpy*(dypz - 2*dzpy) + dypx*dzpy + dxpz*(-dypy - 2*dzpz) + 2*dzpx*dzpz)*pz + (-dxzpz + dzzpx)*pz*pz +
                 py*(dypz*dzpx + dxpz*(-2*dypz + dzpy) + dxpy*(-2*dypy - dzpz) + dypx*(2*dypy + dzpz) + (-dxypz - dxzpy + 2*dyzpx)*pz) +
                 px*(-dxpy*dxpy - dxpz*dxpz + dypx*dypx + dypz*dypz + dzpx*dzpx - 2*dypz*dzpy + dzpy*dzpy + (-dyzpz + dzzpy)*py + (dyypz - dyzpy)*pz)) +
             Kt*((-dxzpz + dzzpx)*py*py + (dxpz*dypy -  dypy*dzpx + dypx*(2*dypz -  dzpy) + dxpy*(-3*dypz + 2*dzpy))*pz + (- dxypy +  dyypx)*pz*pz + py*(-dypz*dzpx + dxpz*(2*dypz - 3*dzpy) + 2*dzpx*dzpy +  dxpy*dzpz -   dypx*dzpz + ( dxypz +  dxzpy - 2*dyzpx)*pz) +
                 px*(-2*dypz*dypz + 4*dypz*dzpy - 2*dzpy*dzpy + ( dyzpz -  dzzpy)*py + (- dyypz + dyzpy)*pz));

        h[y]= Ks*(dxypx + dyypy + dyzpz) +
              Kb*((dxxpy - dxypx)*px*px + (dxxpy - dxypx - dyzpz + dzzpy)*py*py + (dxpz*dypx + dxpy*dzpx - 2*dypx*dzpx + dxpx*(-dypz + dzpy) - 2*dypz*dzpz + 2*dzpy*dzpz)*pz + (-dyzpz + dzzpy)*pz*pz +
                  py*(dxpy*dxpy + dxpz*dxpz - dypx*dypx - dypz*dypz - 2*dxpz*dzpx + dzpx*dzpx + dzpy*dzpy + (dxxpz - dxzpx)*pz) +
                  px*(dxpx*(2*dxpy - 2*dypx) + dypz*dzpx + dxpz*(-2*dypz + dzpy) + dxpy*dzpz - dypx*dzpz + (-dxzpz + dzzpx)*py + (-dxypz + 2*dxzpy - dyzpx)*pz)) +
              Kt*((-dyzpz + dzzpy)*px*px + (-3*dxpz*dypx + dxpy*(2*dxpz - dzpx) + 2*dypx*dzpx + dxpx*(dypz - dzpy))*pz + (dxxpy - dxypx)*pz*pz + py*(-2*dxpz*dxpz + 4*dxpz*dzpx - 2*dzpx*dzpx + (-dxxpz + dxzpx)*pz) +
                  px*(-3*dypz*dzpx + dxpz*(2*dypz - dzpy) + 2*dzpx*dzpy - dxpy*dzpz + dypx*dzpz + (dxzpz - dzzpx)*py + (dxypz - 2*dxzpy + dyzpx)*pz));


        h[z]=Ks*(dxzpx + dyzpy + dzzpz) +
             Kb*((dxxpz - dxzpx)*px*px + (dyypz - dyzpy)*py*py + (dxpy*dxpy + dxpz*dxpz - 2*dxpy*dypx + dypx*dypx + dypz*dypz - dzpx*dzpx - dzpy*dzpy)*pz + (dxxpz - dxzpx + dyypz - dyzpy)*pz*pz +
                 py*(dxpz*dypx + dxpy*dzpx - 2*dypx*dzpx + dypy*(2*dypz - 2*dzpy) + dxpx*(dypz - dzpy) + (dxxpy - dxypx)*pz) +
                 px*(dxpz*dypy + dxpx*(2*dxpz - 2*dzpx) - dypy*dzpx + dxpy*(dypz - 2*dzpy) + dypx*dzpy + (2*dxypz - dxzpy - dyzpx)*py + (-dxypy + dyypx)*pz))+
             Kt*((dyypz - dyzpy)*px*px + (dxxpz - dxzpx)*py*py + (-2*dxpy*dxpy + 4*dxpy*dypx - 2*dypx*dypx)*pz + py*(-dxpz*dypx + dxpy*(2*dxpz - 3*dzpx) + 2*dypx*dzpx + dxpx*(-dypz + dzpy) + (-dxxpy + dxypx)*pz) +
                 px*(-dxpz*dypy + dypy*dzpx + dypx*(2*dypz - 3*dzpy) + dxpy*(-dypz + 2*dzpy) + (-2*dxypz + dxzpy + dyzpx)*py + (dxypy - dyypx)*pz));

        k4[x] = h[x]/gama-nu*(Pol[x]*u[x][x]+Pol[y]*u[x][y]+Pol[z]*u[x][z]) + lambda*Pol[x]/delmu + (W[x][x]*Pol[x]+W[x][y]*Pol[y]+W[x][z]*Pol[z]);
        k4[y] = h[y]/gama-nu*(Pol[x]*u[y][x]+Pol[y]*u[y][y]+Pol[z]*u[y][z]) + lambda*Pol[y]/delmu + (W[y][x]*Pol[x]+W[y][y]*Pol[y]+W[y][z]*Pol[z]);
        k4[z] = h[z]/gama-nu*(Pol[x]*u[z][x]+Pol[y]*u[z][y]+Pol[z]*u[z][z]) + lambda*Pol[z]/delmu + (W[z][x]*Pol[x]+W[z][y]*Pol[y]+W[z][z]*Pol[z]);

        Pol = dPol + (dt / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4);
        for (int j = 0; j < Boundary.size(); j++) {
            auto p = Boundary.get<0>(j);
            Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                         sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                         sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
            Particles.getProp<0>(p)[z] =0;
        }

        k1 = V;
        k2 = 0.5 * dt * k1 + V;
        k3 = 0.5 * dt * k2 + V;
        k4 = dt * k3 + V;
        Pos = Pos + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);

        Particles.map();

        Particles.ghost_get<0, ExtForce, 27>();
        indexUpdate(Particles,Boundary, bulk, up, down, left,right,front,back);
        vector_dist_subset<3, double, aggregate<VectorS<3, double>, VectorS<3, double>, double[3][3], VectorS<3, double>, double, double[3][3], double[3][3], VectorS<3, double>, VectorS<3, double>, VectorS<3, double>, VectorS<3, double>, double, double, double, double, double, double, double, VectorS<3, double>, double, double, double[3], double[3], double[3], double[3], double[3], double[3], double, VectorS<3, double>, VectorS<3, double>, VectorS<3, double>, VectorS<3, double>, double, double, double, double[3][3],double[3][3]>> Particles_subset(
                Particles, bulk);
        auto Pol_bulk = getV<0>(Particles_subset);
        auto P_bulk = getV<Pressure>(Particles_subset);
        auto dPol_bulk = getV<8>(Particles_subset);
        auto RHS_bulk = getV<10>(Particles_subset);

        //Particles_subset.write("debug");

        tt.start();
        Dx.update(Particles);
        Dy.update(Particles);
        Dz.update(Particles);
        Dxy.update(Particles);
        Dxz.update(Particles);
        Dyz.update(Particles);
        auto Dyx = Dxy;
        auto Dzy = Dyz;
        auto Dzx = Dxz;
        Dxx.update(Particles);
        Dyy.update(Particles);
        Dzz.update(Particles);

        Bulk_Dx.update(Particles_subset);
        Bulk_Dy.update(Particles_subset);
        Bulk_Dz.update(Particles_subset);


        tt.stop();
        if (v_cl.rank() == 0) {
            std::cout << "Updation of operators took " << tt.getwct() << " seconds." << std::endl;
            std::cout << "Time step " << ctr - 1 << " : " << tim << " over." << std::endl;
            std::cout << "----------------------------------------------------------" << std::endl;
        }

        tim += dt;
    }

    Particles.deleteGhost();
    Particles.write("Polar_Last");

    Dx.deallocate(Particles);
    Dy.deallocate(Particles);
    Dz.deallocate(Particles);
    Dxy.deallocate(Particles);
    Dxz.deallocate(Particles);
    Dyz.deallocate(Particles);
    Dxx.deallocate(Particles);
    Dyy.deallocate(Particles);
    Dzz.deallocate(Particles);
    Bulk_Dx.deallocate(Particles_subset);
    Bulk_Dy.deallocate(Particles_subset);
    Bulk_Dz.deallocate(Particles_subset);
    Particles.deleteGhost();
    tt2.stop();
    if (v_cl.rank() == 0) {
        std::cout << "The simulation took " << tt2.getcputime() << "(CPU) ------ " << tt2.getwct()
                  << "(Wall) Seconds.";
    }
}


//
// Created by Abhinav Singh on 15.06.20.
//

#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
#define BOOST_MPL_LIMIT_VECTOR_SIZE 40

#include "config.h"

#define BOOST_TEST_DYN_LINK


#include "util/util_debug.hpp"
#include <boost/test/unit_test.hpp>
#include <iostream>
#include "../DCPSE_op.hpp"
#include "../DCPSE_Solver.hpp"
#include "Operators/Vector/vector_dist_operators.hpp"
#include "Vector/vector_dist_subset.hpp"
#include "../EqnsStruct.hpp"
#include "Decomposition/Distribution/SpaceDistribution.hpp"

template<typename particle_type>
void indexUpdate(
        particle_type &Particles,openfpm::vector<aggregate<int>> &Boundary,openfpm::vector<aggregate<int>> &bulk,
        Box<3, double> &up, Box<3, double> &down, Box<3, double> &left, Box<3, double> &right,Box<3, double> &front, Box<3, double> &back) {
    Boundary.clear();
    bulk.clear();

    auto it2 = Particles.getDomainIterator();
    while (it2.isNext()) {
        auto p = it2.get();
        Point<3, double> xp = Particles.getPos(p);
        if (front.isInside(xp) == true) {
            Boundary.add();
            Boundary.last().get<0>() = p.getKey();
        } else if (back.isInside(xp) == true) {
            Boundary.add();
            Boundary.last().get<0>() = p.getKey();
        } else if (left.isInside(xp) == true) {
            Boundary.add();
            Boundary.last().get<0>() = p.getKey();
        } else if (right.isInside(xp) == true) {
            Boundary.add();
            Boundary.last().get<0>() = p.getKey();
        } else if (up.isInside(xp) == true) {
            Boundary.add();
            Boundary.last().get<0>() = p.getKey();
        } else if (down.isInside(xp) == true) {
            Boundary.add();
            Boundary.last().get<0>() = p.getKey();
        } else {
            bulk.add();
            bulk.last().get<0>() = p.getKey();
        }
        ++it2;
    }

}



BOOST_AUTO_TEST_SUITE(dcpse_op_suite_tests3)
    BOOST_AUTO_TEST_CASE(Active3dSimple) {
        timer tt2;
        tt2.start();
        size_t grd_sz = 15;
        double dt = 1e-3;
        double boxsize = 100;
        const size_t sz[3] = {grd_sz, grd_sz, grd_sz};
        Box<3, double> box({0, 0, 0}, {boxsize, boxsize, boxsize});
        size_t bc[3] = {NON_PERIODIC, NON_PERIODIC, NON_PERIODIC};
        double Lx = box.getHigh(0);
        double Ly = box.getHigh(1);
        double Lz = box.getHigh(2);
        double spacing = box.getHigh(0) / (sz[0] - 1);
        double rCut = 3.9 * spacing;
        double rCut2 = 3.9 * spacing;
        int ord = 2;
        int ord2 = 2;
        double sampling_factor = 4.0;
        double sampling_factor2 = 2.4;
        Ghost<3, double> ghost(rCut);
        auto &v_cl = create_vcluster();
        /*                                 pol                          V           vort              Ext           Press     strain       stress            Mfield,            RHS             FE      Q                V_t                 dV                dPol     */
        /*                                 0                          1             2               3               4           5             6                7                8                 9     10               11                  12                13l */
        vector_dist<3, double, aggregate<VectorS<3, double>, VectorS<3, double>, double[3][3], VectorS<3, double>, double, double[3][3], double[3][3], VectorS<3, double>, VectorS<3, double>, double,double[3][3],VectorS<3, double>,VectorS<3, double>,VectorS<3, double>,VectorS<3, double>,VectorS<3, double>,VectorS<3, double>,VectorS<3, double>>> Particles(0, box, bc, ghost);
        Particles.setPropNames({"Polarization","Velocity","Vorticity","External Force","Pressure","Strain-rate","Stress","Molecular Field","Velocity RHS","Franck Energy Density","Q-Tensor","V_t","dV","dPol","k1","k2","k3","k4"});
        double x0, y0, z0, x1, y1, z1;
        x0 = box.getLow(0);
        y0 = box.getLow(1);
        z0 = box.getLow(2);
        x1 = box.getHigh(0);
        y1 = box.getHigh(1);
        z1 = box.getHigh(2);

        auto it = Particles.getGridIterator(sz);
        while (it.isNext()) {
            Particles.add();
            auto key = it.get();
            double x = key.get(0) * it.getSpacing(0);
            Particles.getLastPos()[0] = x;
            double y = key.get(1) * it.getSpacing(1);
            Particles.getLastPos()[1] = y;
            double z = key.get(2) * it.getSpacing(1);
            Particles.getLastPos()[2] = z;
            ++it;
        }

        Particles.map();
        Particles.ghost_get<0>();

        openfpm::vector<aggregate<int>> bulk;
        openfpm::vector<aggregate<int>> Boundary;

        constexpr int x = 0;
        constexpr int y = 1;
        constexpr int z = 2;


        constexpr int Polarization = 0;
        constexpr int Velocity = 1;
        constexpr int Vorticity = 2;
        constexpr int ExtForce = 3;
        constexpr int Pressure = 4;
        constexpr int Strain_rate = 5;
        constexpr int Stress = 6;
        constexpr int MolField = 7;
        auto Pos = getV<PROP_POS>(Particles);

        auto Pol = getV<Polarization>(Particles);
        auto V = getV<Velocity>(Particles);
        auto W = getV<Vorticity>(Particles);
        auto g = getV<ExtForce>(Particles);
        auto P = getV<Pressure>(Particles);
        auto u = getV<Strain_rate>(Particles);
        auto sigma = getV<Stress>(Particles);
        auto h = getV<MolField>(Particles);
        auto RHS = getV<8>(Particles);
        auto FranckEnergyDensity = getV<9>(Particles);
        auto q = getV<10>(Particles);
        auto V_t = getV<11>(Particles);
        auto dV  = getV<12>(Particles);
        auto dPol= getV<13>(Particles);
        auto k1  =getV<14>(Particles);
        auto k2  =getV<15>(Particles);
        auto k3  =getV<16>(Particles);
        auto k4  =getV<17>(Particles);


        texp_v<double> div;
        texp_v<double> delmu;
        /*texp_v<VectorS<3, double>> k1 ;
        texp_v<VectorS<3, double>> k2 ;
        texp_v<VectorS<3, double>> k3 ;
        texp_v<VectorS<3, double>> k4 ;
*/

        double eta = 1.0;
        double nu = -0.5;
        double gama = 0.1;
        double zeta = 0.07;
        double Ks = 1.0;
        double Kt = 1.1;
        double Kb = 1.5;
        //double lambda = 0.1;
        //double delmu = -1.0;
        g = 0;
        P=-1.0;
        delmu = P;
        P = 0;
        V = 0;
        // Here fill up the boxes for particle boundary detection.
        Particles.ghost_get<ExtForce>(SKIP_LABELLING);

        // Here fill up the boxes for particle detection.

        Box<3, double> up(
                {box.getLow(0) - spacing / 2.0, box.getHigh(1) - spacing / 2.0, box.getLow(2) - spacing / 2.0},
                {box.getHigh(0) + spacing / 2.0, box.getHigh(1) + spacing / 2.0, box.getHigh(2) + spacing / 2.0});

        Box<3, double> down(
                {box.getLow(0) - spacing / 2.0, box.getLow(1) - spacing / 2.0, box.getLow(2) - spacing / 2.0},
                {box.getHigh(0) + spacing / 2.0, box.getLow(1) + spacing / 2.0, box.getHigh(2) + spacing / 2.0});

        Box<3, double> left(
                {box.getLow(0) - spacing / 2.0, box.getLow(1) - spacing / 2.0, box.getLow(2) - spacing / 2.0},
                {box.getLow(0) + spacing / 2.0, box.getHigh(1) + spacing / 2.0, box.getHigh(2) + spacing / 2.0});

        Box<3, double> right(
                {box.getHigh(0) - spacing / 2.0, box.getLow(1) - spacing / 2.0, box.getLow(2) - spacing / 2.0},
                {box.getHigh(0) + spacing / 2.0, box.getHigh(1) + spacing / 2.0, box.getHigh(2) + spacing / 2.0});

        Box<3, double> front(
                {box.getLow(0) - spacing / 2.0, box.getLow(1) - spacing / 2.0, box.getLow(2) - spacing / 2.0},
                {box.getHigh(0) + spacing / 2.0, box.getHigh(1) + spacing / 2.0, box.getLow(2) + spacing / 2.0});

        Box<3, double> back(
                {box.getLow(0) - spacing / 2.0, box.getLow(1) - spacing / 2.0, box.getHigh(2) - spacing / 2.0},
                {box.getHigh(0) + spacing / 2.0, box.getHigh(1) + spacing / 2.0, box.getHigh(2) + spacing / 2.0});

        openfpm::vector<Box<3, double>> boxes;
        boxes.add(up);
        boxes.add(down);
        boxes.add(left);
        boxes.add(right);
        boxes.add(front);
        boxes.add(back);
        VTKWriter<openfpm::vector<Box<3, double>>, VECTOR_BOX> vtk_box;
        vtk_box.add(boxes);
        vtk_box.write("boxes_3d.vtk");
        auto it2 = Particles.getDomainIterator();

        while (it2.isNext()) {
            auto p = it2.get();
            Point<3, double> xp = Particles.getPos(p);
            Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * xp[x] - Lx) / Lx) - sin((2 * xp[y] - Ly) / Ly)));
            Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * xp[x] - Lx) / Lx) - sin((2 * xp[y] - Ly) / Ly)));
            Particles.getProp<0>(p)[z] = 0;
            if (front.isInside(xp) == true) {
                Boundary.add();
                Boundary.last().get<0>() = p.getKey();
            } else if (back.isInside(xp) == true) {
                Boundary.add();
                Boundary.last().get<0>() = p.getKey();
            } else if (left.isInside(xp) == true) {
                Boundary.add();
                Boundary.last().get<0>() = p.getKey();
            } else if (right.isInside(xp) == true) {
                Boundary.add();
                Boundary.last().get<0>() = p.getKey();
            } else if (up.isInside(xp) == true) {
                Boundary.add();
                Boundary.last().get<0>() = p.getKey();
            } else if (down.isInside(xp) == true) {
                Boundary.add();
                Boundary.last().get<0>() = p.getKey();
            } else {
                bulk.add();
                bulk.last().get<0>() = p.getKey();
            }
            ++it2;
        }

        vector_dist_subset<3, double, aggregate<VectorS<3, double>, VectorS<3, double>, double[3][3], VectorS<3, double>, double, double[3][3], double[3][3], VectorS<3, double>, VectorS<3, double>, double,double[3][3],VectorS<3, double>,VectorS<3, double>,VectorS<3, double>,VectorS<3, double>,VectorS<3, double>,VectorS<3, double>,VectorS<3, double>>> Particles_subset(Particles, bulk);
        auto Pol_bulk = getV<0>(Particles_subset);
        auto P_bulk = getV<Pressure>(Particles_subset);
        //auto dPol_bulk = getV<8>(Particles_subset);
        auto RHS_bulk = getV<8>(Particles_subset);

        //Particles.write_frame("Active3d_Parts", 0);

        //Particles_subset.write("Pars");
        Derivative_x Dx(Particles, ord, rCut, sampling_factor, support_options::RADIUS), Bulk_Dx(Particles_subset, ord,
                                                                                                 rCut, sampling_factor,
                                                                                                 support_options::RADIUS);
        Derivative_y Dy(Particles, ord, rCut, sampling_factor, support_options::RADIUS), Bulk_Dy(Particles_subset, ord,
                                                                                                 rCut, sampling_factor,
                                                                                                 support_options::RADIUS);
        Derivative_z Dz(Particles, ord, rCut, sampling_factor, support_options::RADIUS), Bulk_Dz(Particles_subset, ord,
                                                                                                 rCut, sampling_factor,
                                                                                                 support_options::RADIUS);
        Derivative_xy Dxy(Particles, ord, rCut2, sampling_factor2, support_options::RADIUS);
        Derivative_yz Dyz(Particles, ord, rCut2, sampling_factor2, support_options::RADIUS);
        Derivative_xz Dxz(Particles, ord, rCut2, sampling_factor2, support_options::RADIUS);
        auto Dyx = Dxy;
        auto Dzy = Dyz;
        auto Dzx = Dxz;

        Derivative_xx Dxx(Particles, ord, rCut2, sampling_factor2,
                          support_options::RADIUS);//, Dxx2(Particles, ord2, rCut2, sampling_factor2, support_options::RADIUS);
        Derivative_yy Dyy(Particles, ord, rCut2, sampling_factor2,
                          support_options::RADIUS);
        Derivative_zz Dzz(Particles, ord, rCut2, sampling_factor2,
                          support_options::RADIUS);

        //V_t = V;
        auto px=Pol[x];
        auto py=Pol[y];
        auto pz=Pol[z];

        Particles.ghost_get<Polarization>(SKIP_LABELLING);
        texp_v<double> dxpx=Dx(Pol[x]), dxpy=Dx(Pol[y]), dxpz=Dx(Pol[z]), dypx=Dy(Pol[x]), dypy=Dy(Pol[y]), dypz=Dy(Pol[z]),dzpx=Dz(Pol[x]),dzpy=Dz(Pol[y]),dzpz=Dz(Pol[z]),
                dxxpx=Dxx(Pol[x]),dxxpy=Dxx(Pol[y]),dxxpz=Dxx(Pol[z]),dyypx=Dyy(Pol[x]), dyypy=Dyy(Pol[y]), dyypz=Dyy(Pol[z]), dzzpx=Dzz(Pol[x]), dzzpy=Dzz(Pol[y]),
                dzzpz=Dzz(Pol[z]), dxypx=Dxy(Pol[x]), dxypy=Dxy(Pol[y]), dxypz=Dxy(Pol[z]), dxzpx=Dxz(Pol[x]), dxzpy=Dxz(Pol[y]), dxzpz=Dxz(Pol[z]), dyzpx=Dyz(Pol[x]), dyzpy=Dyz(Pol[y]),
                dyzpz=Dyz(Pol[z]) , dxhx=Dx(h[x]), dxhy=Dx(h[y]), dxhz=Dx(h[z]), dyhx=Dy(h[x]), dyhy=Dy(h[y]), dyhz=Dy(h[z]), dzhx=Dz(h[x]), dzhy=Dz(h[y]), dzhz=Dz(h[z]),
                dxqxx=Dx(Pol[x]*Pol[x]-1/3.0*(Pol[x]*Pol[x]+Pol[y]*Pol[y]+Pol[z]*Pol[z])),
                dyqxy=Dy(Pol[x]*Pol[x]) , dzqxz=Dz(Pol[x]*Pol[z]) , dxqyx=Dx(Pol[y]*Pol[x]),
                dyqyy=Dy(Pol[y]*Pol[y]-1/3.0*(Pol[x]*Pol[x]+Pol[y]*Pol[y]+Pol[z]*Pol[z])),
                dzqyz=Dz(Pol[y]*Pol[z]) , dxqzx=Dx(Pol[z]*Pol[x]) , dyqzy=Dy(Pol[z]*Pol[y]),
                dzqzz=Dz(Pol[z]*Pol[z]-1/3.0*(Pol[x]*Pol[x]+Pol[y]*Pol[y]+Pol[z]*Pol[z]));

        eq_id vx, vy, vz;

        vx.setId(0);
        vy.setId(1);
        vz.setId(2);
        timer tt;
        double V_err_eps = 5e-2;
        double V_err = 1, V_err_old;
        int n = 0;
        int nmax = 2;
        int ctr = 0, errctr, Vreset = 0;
        double tim = 0;
        double tf = 1.024;
        double sum, sum1, sum_k;
        while (tim <= tf) {
            tt.start();
            petsc_solver<double> solverPetsc;
            solverPetsc.setSolver(KSPGMRES);
            //solverPetsc.setRestart(250);
            solverPetsc.setPreconditioner(PCJACOBI);

            FranckEnergyDensity = 0.5*Ks*(dxpx + dypy + dzpz)*(dxpx + dypy + dzpz) +
                                  0.5*Kt*((dypz - dzpy)*px + (-dxpz + dzpx)*py + (dxpy - dypx)*pz)*((dypz - dzpy)*px + (-dxpz + dzpx)*py + (dxpy - dypx)*pz) +
                                  0.5*Kb*((-dxpz*px + dzpx*px - dypz*py + dzpy*py)*(-dxpz*px + dzpx*px - dypz*py + dzpy*py) +
                                          (dxpy*py - dypx*py + dxpz*pz - dzpx*pz)*(dxpy*py - dypx*py + dxpz*pz - dzpx*pz) +
                                          (-dxpy*px + dypx*px + dypz*pz - dzpy*pz)*(-dxpy*px + dypx*px + dypz*pz - dzpy*pz));

            h[x]=Ks*(dxxpx + dxypy + dxzpz) +
                 Kb*((-dxypy - dxzpz + dyypx + dzzpx)*px*px + (-dxypy + dyypx)*py*py + (dypy*dzpx + dxpy*(dypz - 2*dzpy) + dypx*dzpy + dxpz*(-dypy - 2*dzpz) + 2*dzpx*dzpz)*pz + (-dxzpz + dzzpx)*pz*pz +
                     py*(dypz*dzpx + dxpz*(-2*dypz + dzpy) + dxpy*(-2*dypy - dzpz) + dypx*(2*dypy + dzpz) + (-dxypz - dxzpy + 2*dyzpx)*pz) +
                     px*(-dxpy*dxpy - dxpz*dxpz + dypx*dypx + dypz*dypz + dzpx*dzpx - 2*dypz*dzpy + dzpy*dzpy + (-dyzpz + dzzpy)*py + (dyypz - dyzpy)*pz)) +
                 Kt*((-dxzpz + dzzpx)*py*py + (dxpz*dypy -  dypy*dzpx + dypx*(2*dypz -  dzpy) + dxpy*(-3*dypz + 2*dzpy))*pz + (- dxypy +  dyypx)*pz*pz + py*(-dypz*dzpx + dxpz*(2*dypz - 3*dzpy) + 2*dzpx*dzpy +  dxpy*dzpz -   dypx*dzpz + ( dxypz +  dxzpy - 2*dyzpx)*pz) +
                     px*(-2*dypz*dypz + 4*dypz*dzpy - 2*dzpy*dzpy + ( dyzpz -  dzzpy)*py + (- dyypz + dyzpy)*pz));

            h[y]= Ks*(dxypx + dyypy + dyzpz) +
                  Kb*((dxxpy - dxypx)*px*px + (dxxpy - dxypx - dyzpz + dzzpy)*py*py + (dxpz*dypx + dxpy*dzpx - 2*dypx*dzpx + dxpx*(-dypz + dzpy) - 2*dypz*dzpz + 2*dzpy*dzpz)*pz + (-dyzpz + dzzpy)*pz*pz +
                      py*(dxpy*dxpy + dxpz*dxpz - dypx*dypx - dypz*dypz - 2*dxpz*dzpx + dzpx*dzpx + dzpy*dzpy + (dxxpz - dxzpx)*pz) +
                      px*(dxpx*(2*dxpy - 2*dypx) + dypz*dzpx + dxpz*(-2*dypz + dzpy) + dxpy*dzpz - dypx*dzpz + (-dxzpz + dzzpx)*py + (-dxypz + 2*dxzpy - dyzpx)*pz)) +
                  Kt*((-dyzpz + dzzpy)*px*px + (-3*dxpz*dypx + dxpy*(2*dxpz - dzpx) + 2*dypx*dzpx + dxpx*(dypz - dzpy))*pz + (dxxpy - dxypx)*pz*pz + py*(-2*dxpz*dxpz + 4*dxpz*dzpx - 2*dzpx*dzpx + (-dxxpz + dxzpx)*pz) +
                      px*(-3*dypz*dzpx + dxpz*(2*dypz - dzpy) + 2*dzpx*dzpy - dxpy*dzpz + dypx*dzpz + (dxzpz - dzzpx)*py + (dxypz - 2*dxzpy + dyzpx)*pz));


            h[z]=Ks*(dxzpx + dyzpy + dzzpz) +
                 Kb*((dxxpz - dxzpx)*px*px + (dyypz - dyzpy)*py*py + (dxpy*dxpy + dxpz*dxpz - 2*dxpy*dypx + dypx*dypx + dypz*dypz - dzpx*dzpx - dzpy*dzpy)*pz + (dxxpz - dxzpx + dyypz - dyzpy)*pz*pz +
                     py*(dxpz*dypx + dxpy*dzpx - 2*dypx*dzpx + dypy*(2*dypz - 2*dzpy) + dxpx*(dypz - dzpy) + (dxxpy - dxypx)*pz) +
                     px*(dxpz*dypy + dxpx*(2*dxpz - 2*dzpx) - dypy*dzpx + dxpy*(dypz - 2*dzpy) + dypx*dzpy + (2*dxypz - dxzpy - dyzpx)*py + (-dxypy + dyypx)*pz))+
                 Kt*((dyypz - dyzpy)*px*px + (dxxpz - dxzpx)*py*py + (-2*dxpy*dxpy + 4*dxpy*dypx - 2*dypx*dypx)*pz + py*(-dxpz*dypx + dxpy*(2*dxpz - 3*dzpx) + 2*dypx*dzpx + dxpx*(-dypz + dzpy) + (-dxxpy + dxypx)*pz) +
                     px*(-dxpz*dypy + dypy*dzpx + dypx*(2*dypz - 3*dzpy) + dxpy*(-dypz + 2*dzpy) + (-2*dxypz + dxzpy + dyzpx)*py + (dxypy - dyypx)*pz));


            sigma[x][x] =
                    -dxpx*(dxpx + dypy + dzpz)*Ks -
                    dxpy*Kt*pz*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) -
                    dxpz*Kt*py*(-dypz*px + dzpy*px + dxpz*py - dzpx*py - dxpy*pz + dypx*pz) -
                    0.5*dxpz*Kb*(2*px*(dxpz*px - dzpx*px + (dypz - dzpy)*py) + 2*pz*(dxpy*py - dypx*py + (dxpz - dzpx)*pz)) -
                    0.5*dxpy*Kb*(2*py*(dxpy*py - dypx*py + (dxpz - dzpx)*pz) + 2*px*(dxpy*px - dypx*px + (-dypz + dzpy)*pz));
            sigma[x][y] =
                    -dxpy*(dxpx + dypy + dzpz)*Ks - dxpz*Kb*py*(dxpz*px - dzpx*px + (dypz - dzpy)*py) +
                    dxpx*Kt*pz*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) +
                    dxpz*Kt*px*(-dypz*px + dzpy*px + dxpz*py - dzpx*py - dxpy*pz + dypx*pz) +
                    dxpx*Kb*py*(dxpy*py - dypx*py + (dxpz - dzpx)*pz) -
                    dxpz*Kb*pz*(-dxpy*px + dypx*px + (dypz - dzpy)*pz) +
                    dxpx*Kb*px*(dxpy*px - dypx*px + (-dypz + dzpy)*pz);

            sigma[x][z] =
                    -dxpz*(dxpx + dypy + dzpz)*Ks +
                    dxpy*Kt*px*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) +
                    dxpx*Kt*py*(-dypz*px + dzpy*px + dxpz*py - dzpx*py - dxpy*pz + dypx*pz) -
                    0.5*dxpx*Kb*(2*px*(-dxpz*px + dzpx*px + (-dypz + dzpy)*py) - 2*pz*(dxpy*py - dypx*py + (dxpz - dzpx)*pz)) -
                    0.5*dxpy*Kb*(2*py*(-dxpz*px + dzpx*px + (-dypz + dzpy)*py) - 2*pz*(-dxpy*px + dypx*px + (dypz - dzpy)*pz));
            sigma[y][x] =
                    -dypx*(dxpx + dypy + dzpz)*Ks +
                    dypz*Kt*py*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) -
                    dypy*Kt*pz*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) -
                    0.5*dypz*Kb*(2*px*(dxpz*px - dzpx*px + (dypz - dzpy)*py) + 2*pz*(dxpy*py - dypx*py + (dxpz - dzpx)*pz)) -
                    0.5*dypy*Kb*(2*py*(dxpy*py - dypx*py + (dxpz - dzpx)*pz) + 2*px*(dxpy*px - dypx*px + (-dypz + dzpy)*pz));
            sigma[y][y] =
                    -dypy*(dxpx + dypy + dzpz)*Ks -
                    dypz*Kb*py*(dxpz*px - dzpx*px + (dypz - dzpy)*py) -
                    dypz*Kt*px*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) -
                    dypx*Kt*pz*(-dypz*px + dzpy*px + dxpz*py - dzpx*py -dxpy*pz + dypx*pz) -
                    dypx*Kb*py*(-dxpy*py + dypx*py + (-dxpz + dzpx)*pz) -
                    dypx*Kb*px*(-dxpy*px + dypx*px + (dypz - dzpy)*pz) -
                    dypz*Kb*pz*(-dxpy*px + dypx*px + (dypz - dzpy)*pz);
            sigma[y][z] =
                    -dypz*(dxpx + dypy + dzpz)*Ks +
                    dypy*Kt*px*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) +
                    dypx*Kt*py*(-dypz*px + dzpy*px + dxpz*py - dzpx*py -dxpy*pz + dypx*pz) -
                    0.5*dypx*Kb*(2*px*(-dxpz*px + dzpx*px + (-dypz + dzpy)*py) - 2*pz*(dxpy*py - dypx*py + (dxpz - dzpx)*pz)) -
                    0.5*dypy*Kb*(2*py*(-dxpz*px + dzpx*px + (-dypz + dzpy)*py) - 2*pz*(-dxpy*px + dypx*px + (dypz - dzpy)*pz));
            sigma[z][x] =
                    -dzpx*(dxpx + dypy + dzpz)*Ks -
                    dzpz*Kt*py*(-dypz*px + dzpy*px + dxpz*py - dzpx*py - dxpy*pz + dypx*pz) +
                    dzpy*Kt*pz*(-dypz*px + dzpy*px + dxpz*py - dzpx*py - dxpy*pz + dypx*pz) -
                    0.5*dzpz*Kb*(2*px*(dxpz*px - dzpx*px + (dypz - dzpy)*py) + 2*pz*(dxpy*py - dypx*py + (dxpz - dzpx)*pz)) -
                    0.5*dzpy*Kb*(2*py*(dxpy*py - dypx*py + (dxpz - dzpx)*pz) + 2*px*(dxpy*px - dypx*px + (-dypz + dzpy)*pz));
            sigma[z][y] =
                    -dzpy*(dxpx + dypy + dzpz)*Ks -
                    dzpz*Kb*py*(dxpz*px - dzpx*px + (dypz - dzpy)*py) -
                    dzpz*Kt*px*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) +
                    dzpx*Kt*pz*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) -
                    dzpx*Kb*py*(-dxpy*py + dypx*py + (-dxpz + dzpx)*pz) -
                    dzpz*Kb*pz*(-dxpy*px + dypx*px + (dypz - dzpy)*pz) +
                    dzpx*Kb*px*(dxpy*px - dypx*px + (-dypz + dzpy)*pz);

            sigma[z][z] =
                    -dzpz*(dxpx + dypy + dzpz)*Ks -
                    dzpx*Kt*py*(dypz*px - dzpy*px - dxpz*py + dzpx*py + dxpy*pz - dypx*pz) -
                    dzpy*Kt*px*(-dypz*px + dzpy*px + dxpz*py - dzpx*py - dxpy*pz + dypx*pz) -
                    0.5*dzpx*Kb*(2*px*(-dxpz*px + dzpx*px + (-dypz + dzpy)*py) - 2*pz*(dxpy*py - dypx*py + (dxpz - dzpx)*pz)) -
                    0.5*dzpy*Kb*(2*py*(-dxpz*px + dzpx*px + (-dypz + dzpy)*py) - 2*pz*(-dxpy*px + dypx*px + (dypz - dzpy)*pz));


            Particles.ghost_get<Stress>(SKIP_LABELLING);
            Particles.ghost_get<MolField>(SKIP_LABELLING);
            dxhx=Dx(h[x]);
            dxhy=Dx(h[y]);
            dxhz=Dx(h[z]);
            dyhx=Dy(h[x]);
            dyhy=Dy(h[y]);
            dyhz=Dy(h[z]);
            dzhx=Dz(h[x]);
            dzhy=Dz(h[y]);
            dzhz=Dz(h[z]);

            dV[x] = 0.5*(-dypy*h[x] + dypx*h[y] + dyhy*px - dyhx*py) + 0.5*(-dzpz*h[x] + dzpx*h[z] + dzhz*px - dzhx*pz) +
                    zeta*delmu*(dxqxx + dyqxy + dzqxz)  +
                    nu*(1/2*(-dypy*h[x] + dypx*h[y] + dyhy*px - dyhx*py) + 1/3*(-dxpx*h[x] - dxpy*h[y] - dxpz*h[z] - dxhx*px - dxhy*py -dxhz*pz) + 1/2*(-dzpz*h[x] + dzpx*h[z] + dzhz*px - dzhx*pz))+
                    Dx(sigma[x][x])+Dy(sigma[x][y])+Dz(sigma[x][z]);


            dV[y] = 0.5*(dxpy*h[x] - dxpx*h[y] - dxhy*px + dxhx*py) + 0.5*(-dzpz*h[y] + dzpy*h[z] + dzhz*py - dzhy*pz) +
                    zeta*delmu*(dxqyx + dyqyy + dzqyz) +
                    nu*(1/2*(dxpy*h[x] - dxpx*h[y] - dxhy*px + dxhx*py) + 1/3*(-dypx*h[x] - dypy*h[y] - dypz*h[z] - dyhx*px - dyhy*py - dyhz*pz) + 1/2*(-dzpz*h[y] + dzpy*h[z] + dzhz*py - dzhy*pz))+
                    Dx(sigma[y][x])+Dy(sigma[y][y])+Dz(sigma[y][z]);

            dV[z]=0.5*(dxpz*h[x] - dxpx*h[z] - dxhz*px + dxhx*pz) + 0.5*(dypz*h[y] - dypy*h[z] - dyhz*py + dyhy*pz) +
                  zeta*delmu*(dxqzx + dyqzy +dzqzz)  +
                  nu*(1/2*(dxpz*h[x] - dxpx*h[z] - dxhz*px + dxhx*pz)+ 1/2*(dypz*h[y] - dypy*h[z] - dyhz*py + dyhy*pz) + 1/3*(-dzpx*h[x] - dzpy*h[y] - dzpz*h[z] - dzhx*px - dzhy*py -dzhz*pz))+
                  Dx(sigma[z][x])+Dy(sigma[z][y])+Dz(sigma[z][z]);

            Particles.ghost_get<8>(SKIP_LABELLING);


            //Particles.write("PolarI");
            //Velocity Solution n iterations



            auto Stokes1 = eta * (2*Dxx(V[x]) + Dxy(V[y]) + Dyy(V[x]) + Dxz(V[z]) + Dzz(V[x]));
            auto Stokes2 = eta * (2*Dyy(V[y]) + Dxx(V[y]) + Dxy(V[x]) + Dyz(V[z]) + Dzz(V[y]));
            auto Stokes3 = eta * (2*Dzz(V[z]) + Dxx(V[z]) + Dxz(V[x]) + Dyy(V[z]) + Dyz(V[y]));
            std::cout << "Init of Velocity took " << tt.getwct() << " seconds." << std::endl;
            tt.start();
            V_err = 1;
            n = 0;
            errctr = 0;
            if (Vreset == 1) {
                P_bulk = 0;
                P = 0;
                Vreset = 0;
            }
            P = 0;
            //Particles.save("PolarCrash_" + std::to_string(ctr));
            while (V_err >= V_err_eps && n <= nmax) {
                Particles.ghost_get<4>(SKIP_LABELLING);
                RHS_bulk[x] = -dV[x]+Bulk_Dx(P);
                RHS_bulk[y] = -dV[y]+Bulk_Dy(P);
                RHS_bulk[z] = -dV[z]+Bulk_Dz(P);
                Particles.ghost_get<10>(SKIP_LABELLING);
                DCPSE_scheme<equations3d3, decltype(Particles)> Solver(Particles);
                Solver.impose(Stokes1, bulk, RHS[0], vx);
                Solver.impose(Stokes2, bulk, RHS[1], vy);
                Solver.impose(Stokes3, bulk, RHS[2], vz);
                Solver.impose(V[x], Boundary, 0, vx);
                Solver.impose(V[y], Boundary, 0, vy);
                Solver.impose(V[z], Boundary, 0, vx);
                /* auto &A=Solver.getA(options_solver::STANDARD);
                 A.getMatrixTriplets().save("ATripletes" + std::to_string(n));
                 auto &B=Solver.getB(options_solver::STANDARD);
                 B.getVec().save("B"+ std::to_string(n));*//*

                                 PetscViewer viewer;
                auto &VA=Solver.getA(options_solver::STANDARD).getMat();
                auto &Vv=Solver.getB(options_solver::STANDARD).getVec();

                char int_string[4];
                sprintf(int_string, "%d", n);
                char data_string[12] = "data_";
                strcat(data_string, int_string);

                PetscViewerBinaryOpen(MPI_COMM_WORLD, data_string, FILE_MODE_WRITE, &viewer);
                *//**//* Save matrix/vector *//**//*
                MatView(VA, viewer); VecView(Vv, viewer);
                PetscViewerDestroy(&viewer);*/
                //MatDestroy(&VA); VecDestroy(&Vv);

                Solver.solve_with_solver(solverPetsc, V[x], V[y], V[z]);
                //Solver.solve(V[x], V[y]);
                Particles.ghost_get<Velocity>(SKIP_LABELLING);
                div = -(Dx(V[x]) + Dy(V[y])+Dz(V[z]));
                P_bulk = P + div;
                sum = 0;
                sum1 = 0;
                for (int j = 0; j < bulk.size(); j++) {
                    auto p = bulk.get<0>(j);
                    sum += (Particles.getProp<11>(p)[0] - Particles.getProp<1>(p)[0]) *
                           (Particles.getProp<11>(p)[0] - Particles.getProp<1>(p)[0]) +
                           (Particles.getProp<11>(p)[1] - Particles.getProp<1>(p)[1]) *
                           (Particles.getProp<11>(p)[1] - Particles.getProp<1>(p)[1]) +
                           (Particles.getProp<11>(p)[2] - Particles.getProp<1>(p)[2]) *
                           (Particles.getProp<11>(p)[2] - Particles.getProp<1>(p)[2]);
                    sum1 += Particles.getProp<1>(p)[0] * Particles.getProp<1>(p)[0] +
                            Particles.getProp<1>(p)[1] * Particles.getProp<1>(p)[1]+
                            Particles.getProp<1>(p)[2] * Particles.getProp<1>(p)[2];
                }
                sum = sqrt(sum);
                sum1 = sqrt(sum1);

                v_cl.sum(sum);
                v_cl.sum(sum1);
                v_cl.execute();
                V_t = V;
                Particles.ghost_get<1, 4, 11>(SKIP_LABELLING);
                V_err_old = V_err;
                V_err = sum / sum1;
                if (V_err > V_err_old || abs(V_err_old - V_err) < 1e-8) {
                    errctr++;
                    //alpha_P -= 0.1;
                } else {
                    errctr = 0;
                }
                if (n > 3) {
                    if (errctr > 3) {
                        std::cout << "CONVERGENCE LOOP BROKEN DUE TO INCREASE/VERY SLOW DECREASE IN ERROR" << std::endl;
                        Vreset = 1;
                        break;
                    } else {
                        Vreset = 0;
                    }
                }
                n++;
                //Particles.write_frame("V_debug", n);
                if (v_cl.rank() == 0) {
                    std::cout << "Rel l2 cgs err in V = " << V_err << " at " << n << std::endl;
                }
            }
            tt.stop();
            u[x][x] = Dx(V[x]);
            u[x][y] = 0.5 * (Dx(V[y]) + Dy(V[x]));
            u[x][z] = 0.5 * (Dx(V[z]) + Dz(V[x]));
            u[y][x] = 0.5 * (Dy(V[x]) + Dx(V[y]));
            u[y][y] = Dy(V[y]);
            u[y][z] = 0.5 * (Dy(V[z]) + Dz(V[y]));
            u[z][x] = 0.5 * (Dz(V[x]) + Dx(V[z]));
            u[z][y] = 0.5 * (Dz(V[y]) + Dy(V[z]));
            u[z][z] = Dz(V[z]);


            //Adaptive CFL
            /*sum=0;
            auto it2 = Particles.getDomainIterator();
            while (it2.isNext()) {
                auto p = it2.get();
                sum += Particles.getProp<Strain_rate>(p)[x][x] * Particles.getProp<Strain_rate>(p)[x][x] +
                        Particles.getProp<Strain_rate>(p)[y][y] * Particles.getProp<Strain_rate>(p)[y][y];
                ++it2;
            }
            sum = sqrt(sum);
            v_cl.sum(sum);
            v_cl.execute();
            dt=0.5/sum;*/
            if (v_cl.rank() == 0) {
                std::cout << "Rel l2 cgs err in V = " << V_err << " and took " << tt.getwct() << " seconds with " << n
                          << " iterations. dt is set to " << dt
                          << std::endl;
            }

            W[x][x] = 0;
            W[x][y] = 0.5 * (Dy(V[x]) - Dx(V[y]));
            W[x][z] = 0.5 * (Dz(V[x]) - Dx(V[z]));
            W[y][x] = 0.5 * (Dx(V[y]) - Dy(V[x]));
            W[y][y] = 0;
            W[y][z] = 0.5 * (Dz(V[y]) - Dy(V[z]));
            W[z][x] = 0.5 * (Dx(V[z]) - Dz(V[x]));
            W[z][y] = 0.5 * (Dy(V[z]) - Dz(V[y]));
            W[z][z] = 0;

            Particles.deleteGhost();
            Particles.write_frame("Polar3d", ctr,BINARY);
            Particles.ghost_get<0>();
            ctr++;
            auto lambda = -1/(3*gama)*(-3*h[x]*Pol[x]-3*h[y]*Pol[y]-3*h[z]*Pol[z]+
                                       gama*nu*(Pol[x]*Pol[x]*u[x][x]+Pol[y]*Pol[y]*u[y][y]+Pol[z]*Pol[z]*u[z][z]+
                                                Pol[x]*(Pol[y]*(u[x][y]+u[y][x])+Pol[z]*(u[x][z]+u[z][x])) +Pol[y]*Pol[z]*(u[y][z]+u[z][y])))/(Pol[x]*Pol[x]+Pol[y]*Pol[y]+Pol[z]*Pol[z]);
            dPol=Pol;
            Particles.ghost_get<8>(SKIP_LABELLING);
            k1[x] = h[x]/gama-nu*(Pol[x]*u[x][x]+Pol[y]*u[x][y]+Pol[z]*u[x][z]) + lambda*Pol[x]/delmu + (W[x][x]*Pol[x]+W[x][y]*Pol[y]+W[x][z]*Pol[z]);
            k1[y] = h[y]/gama-nu*(Pol[x]*u[y][x]+Pol[y]*u[y][y]+Pol[z]*u[y][z]) + lambda*Pol[y]/delmu + (W[y][x]*Pol[x]+W[y][y]*Pol[y]+W[y][z]*Pol[z]);
            k1[z] = h[z]/gama-nu*(Pol[x]*u[z][x]+Pol[y]*u[z][y]+Pol[z]*u[z][z]) + lambda*Pol[z]/delmu + (W[z][x]*Pol[x]+W[z][y]*Pol[y]+W[z][z]*Pol[z]);
            Pol_bulk = dPol + (0.5 * dt) * k1;

            Particles.ghost_get<0>(SKIP_LABELLING);

            dxpx=Dx(Pol[x]);
            dxpy=Dx(Pol[y]);
            dxpz=Dx(Pol[z]);
            dypx=Dy(Pol[x]);
            dypy=Dy(Pol[y]);
            dypz=Dy(Pol[z]);
            dzpx=Dz(Pol[x]);
            dzpy=Dz(Pol[y]);
            dzpz=Dz(Pol[z]);
            dxxpx=Dxx(Pol[x]);
            dxxpy=Dxx(Pol[y]);
            dxxpz=Dxx(Pol[z]);
            dyypx=Dyy(Pol[x]);
            dyypy=Dyy(Pol[y]);
            dyypz=Dyy(Pol[z]);
            dzzpx=Dzz(Pol[x]);
            dzzpy=Dzz(Pol[y]);
            dzzpz=Dzz(Pol[z]);
            dxypx=Dxy(Pol[x]);
            dxypy=Dxy(Pol[y]);
            dxypz=Dxy(Pol[z]);
            dxzpx=Dxz(Pol[x]);
            dxzpy=Dxz(Pol[y]);
            dxzpz=Dxz(Pol[z]);
            dyzpx=Dyz(Pol[x]);
            dyzpy=Dyz(Pol[y]);
            dyzpz=Dyz(Pol[z]);



            h[x]=Ks*(dxxpx + dxypy + dxzpz) +
                 Kb*((-dxypy - dxzpz + dyypx + dzzpx)*px*px + (-dxypy + dyypx)*py*py + (dypy*dzpx + dxpy*(dypz - 2*dzpy) + dypx*dzpy + dxpz*(-dypy - 2*dzpz) + 2*dzpx*dzpz)*pz + (-dxzpz + dzzpx)*pz*pz +
                     py*(dypz*dzpx + dxpz*(-2*dypz + dzpy) + dxpy*(-2*dypy - dzpz) + dypx*(2*dypy + dzpz) + (-dxypz - dxzpy + 2*dyzpx)*pz) +
                     px*(-dxpy*dxpy - dxpz*dxpz + dypx*dypx + dypz*dypz + dzpx*dzpx - 2*dypz*dzpy + dzpy*dzpy + (-dyzpz + dzzpy)*py + (dyypz - dyzpy)*pz)) +
                 Kt*((-dxzpz + dzzpx)*py*py + (dxpz*dypy -  dypy*dzpx + dypx*(2*dypz -  dzpy) + dxpy*(-3*dypz + 2*dzpy))*pz + (- dxypy +  dyypx)*pz*pz + py*(-dypz*dzpx + dxpz*(2*dypz - 3*dzpy) + 2*dzpx*dzpy +  dxpy*dzpz -   dypx*dzpz + ( dxypz +  dxzpy - 2*dyzpx)*pz) +
                     px*(-2*dypz*dypz + 4*dypz*dzpy - 2*dzpy*dzpy + ( dyzpz -  dzzpy)*py + (- dyypz + dyzpy)*pz));

            h[y]= Ks*(dxypx + dyypy + dyzpz) +
                  Kb*((dxxpy - dxypx)*px*px + (dxxpy - dxypx - dyzpz + dzzpy)*py*py + (dxpz*dypx + dxpy*dzpx - 2*dypx*dzpx + dxpx*(-dypz + dzpy) - 2*dypz*dzpz + 2*dzpy*dzpz)*pz + (-dyzpz + dzzpy)*pz*pz +
                      py*(dxpy*dxpy + dxpz*dxpz - dypx*dypx - dypz*dypz - 2*dxpz*dzpx + dzpx*dzpx + dzpy*dzpy + (dxxpz - dxzpx)*pz) +
                      px*(dxpx*(2*dxpy - 2*dypx) + dypz*dzpx + dxpz*(-2*dypz + dzpy) + dxpy*dzpz - dypx*dzpz + (-dxzpz + dzzpx)*py + (-dxypz + 2*dxzpy - dyzpx)*pz)) +
                  Kt*((-dyzpz + dzzpy)*px*px + (-3*dxpz*dypx + dxpy*(2*dxpz - dzpx) + 2*dypx*dzpx + dxpx*(dypz - dzpy))*pz + (dxxpy - dxypx)*pz*pz + py*(-2*dxpz*dxpz + 4*dxpz*dzpx - 2*dzpx*dzpx + (-dxxpz + dxzpx)*pz) +
                      px*(-3*dypz*dzpx + dxpz*(2*dypz - dzpy) + 2*dzpx*dzpy - dxpy*dzpz + dypx*dzpz + (dxzpz - dzzpx)*py + (dxypz - 2*dxzpy + dyzpx)*pz));


            h[z]=Ks*(dxzpx + dyzpy + dzzpz) +
                 Kb*((dxxpz - dxzpx)*px*px + (dyypz - dyzpy)*py*py + (dxpy*dxpy + dxpz*dxpz - 2*dxpy*dypx + dypx*dypx + dypz*dypz - dzpx*dzpx - dzpy*dzpy)*pz + (dxxpz - dxzpx + dyypz - dyzpy)*pz*pz +
                     py*(dxpz*dypx + dxpy*dzpx - 2*dypx*dzpx + dypy*(2*dypz - 2*dzpy) + dxpx*(dypz - dzpy) + (dxxpy - dxypx)*pz) +
                     px*(dxpz*dypy + dxpx*(2*dxpz - 2*dzpx) - dypy*dzpx + dxpy*(dypz - 2*dzpy) + dypx*dzpy + (2*dxypz - dxzpy - dyzpx)*py + (-dxypy + dyypx)*pz))+
                 Kt*((dyypz - dyzpy)*px*px + (dxxpz - dxzpx)*py*py + (-2*dxpy*dxpy + 4*dxpy*dypx - 2*dypx*dypx)*pz + py*(-dxpz*dypx + dxpy*(2*dxpz - 3*dzpx) + 2*dypx*dzpx + dxpx*(-dypz + dzpy) + (-dxxpy + dxypx)*pz) +
                     px*(-dxpz*dypy + dypy*dzpx + dypx*(2*dypz - 3*dzpy) + dxpy*(-dypz + 2*dzpy) + (-2*dxypz + dxzpy + dyzpx)*py + (dxypy - dyypx)*pz));


            k2[x] = h[x]/gama-nu*(Pol[x]*u[x][x]+Pol[y]*u[x][y]+Pol[z]*u[x][z]) + lambda*Pol[x]/delmu + (W[x][x]*Pol[x]+W[x][y]*Pol[y]+W[x][z]*Pol[z]);
            k2[y] = h[y]/gama-nu*(Pol[x]*u[y][x]+Pol[y]*u[y][y]+Pol[z]*u[y][z]) + lambda*Pol[y]/delmu + (W[y][x]*Pol[x]+W[y][y]*Pol[y]+W[y][z]*Pol[z]);
            k2[z] = h[z]/gama-nu*(Pol[x]*u[z][x]+Pol[y]*u[z][y]+Pol[z]*u[z][z]) + lambda*Pol[z]/delmu + (W[z][x]*Pol[x]+W[z][y]*Pol[y]+W[z][z]*Pol[z]);
            Pol_bulk = dPol + (0.5 * dt) * k2;
            Particles.ghost_get<0>(SKIP_LABELLING);
            dxpx=Dx(Pol[x]);
            dxpy=Dx(Pol[y]);
            dxpz=Dx(Pol[z]);
            dypx=Dy(Pol[x]);
            dypy=Dy(Pol[y]);
            dypz=Dy(Pol[z]);
            dzpx=Dz(Pol[x]);
            dzpy=Dz(Pol[y]);
            dzpz=Dz(Pol[z]);
            dxxpx=Dxx(Pol[x]);
            dxxpy=Dxx(Pol[y]);
            dxxpz=Dxx(Pol[z]);
            dyypx=Dyy(Pol[x]);
            dyypy=Dyy(Pol[y]);
            dyypz=Dyy(Pol[z]);
            dzzpx=Dzz(Pol[x]);
            dzzpy=Dzz(Pol[y]);
            dzzpz=Dzz(Pol[z]);
            dxypx=Dxy(Pol[x]);
            dxypy=Dxy(Pol[y]);
            dxypz=Dxy(Pol[z]);
            dxzpx=Dxz(Pol[x]);
            dxzpy=Dxz(Pol[y]);
            dxzpz=Dxz(Pol[z]);
            dyzpx=Dyz(Pol[x]);
            dyzpy=Dyz(Pol[y]);
            dyzpz=Dyz(Pol[z]);

            h[x]=Ks*(dxxpx + dxypy + dxzpz) +
                 Kb*((-dxypy - dxzpz + dyypx + dzzpx)*px*px + (-dxypy + dyypx)*py*py + (dypy*dzpx + dxpy*(dypz - 2*dzpy) + dypx*dzpy + dxpz*(-dypy - 2*dzpz) + 2*dzpx*dzpz)*pz + (-dxzpz + dzzpx)*pz*pz +
                     py*(dypz*dzpx + dxpz*(-2*dypz + dzpy) + dxpy*(-2*dypy - dzpz) + dypx*(2*dypy + dzpz) + (-dxypz - dxzpy + 2*dyzpx)*pz) +
                     px*(-dxpy*dxpy - dxpz*dxpz + dypx*dypx + dypz*dypz + dzpx*dzpx - 2*dypz*dzpy + dzpy*dzpy + (-dyzpz + dzzpy)*py + (dyypz - dyzpy)*pz)) +
                 Kt*((-dxzpz + dzzpx)*py*py + (dxpz*dypy -  dypy*dzpx + dypx*(2*dypz -  dzpy) + dxpy*(-3*dypz + 2*dzpy))*pz + (- dxypy +  dyypx)*pz*pz + py*(-dypz*dzpx + dxpz*(2*dypz - 3*dzpy) + 2*dzpx*dzpy +  dxpy*dzpz -   dypx*dzpz + ( dxypz +  dxzpy - 2*dyzpx)*pz) +
                     px*(-2*dypz*dypz + 4*dypz*dzpy - 2*dzpy*dzpy + ( dyzpz -  dzzpy)*py + (- dyypz + dyzpy)*pz));

            h[y]= Ks*(dxypx + dyypy + dyzpz) +
                  Kb*((dxxpy - dxypx)*px*px + (dxxpy - dxypx - dyzpz + dzzpy)*py*py + (dxpz*dypx + dxpy*dzpx - 2*dypx*dzpx + dxpx*(-dypz + dzpy) - 2*dypz*dzpz + 2*dzpy*dzpz)*pz + (-dyzpz + dzzpy)*pz*pz +
                      py*(dxpy*dxpy + dxpz*dxpz - dypx*dypx - dypz*dypz - 2*dxpz*dzpx + dzpx*dzpx + dzpy*dzpy + (dxxpz - dxzpx)*pz) +
                      px*(dxpx*(2*dxpy - 2*dypx) + dypz*dzpx + dxpz*(-2*dypz + dzpy) + dxpy*dzpz - dypx*dzpz + (-dxzpz + dzzpx)*py + (-dxypz + 2*dxzpy - dyzpx)*pz)) +
                  Kt*((-dyzpz + dzzpy)*px*px + (-3*dxpz*dypx + dxpy*(2*dxpz - dzpx) + 2*dypx*dzpx + dxpx*(dypz - dzpy))*pz + (dxxpy - dxypx)*pz*pz + py*(-2*dxpz*dxpz + 4*dxpz*dzpx - 2*dzpx*dzpx + (-dxxpz + dxzpx)*pz) +
                      px*(-3*dypz*dzpx + dxpz*(2*dypz - dzpy) + 2*dzpx*dzpy - dxpy*dzpz + dypx*dzpz + (dxzpz - dzzpx)*py + (dxypz - 2*dxzpy + dyzpx)*pz));


            h[z]=Ks*(dxzpx + dyzpy + dzzpz) +
                 Kb*((dxxpz - dxzpx)*px*px + (dyypz - dyzpy)*py*py + (dxpy*dxpy + dxpz*dxpz - 2*dxpy*dypx + dypx*dypx + dypz*dypz - dzpx*dzpx - dzpy*dzpy)*pz + (dxxpz - dxzpx + dyypz - dyzpy)*pz*pz +
                     py*(dxpz*dypx + dxpy*dzpx - 2*dypx*dzpx + dypy*(2*dypz - 2*dzpy) + dxpx*(dypz - dzpy) + (dxxpy - dxypx)*pz) +
                     px*(dxpz*dypy + dxpx*(2*dxpz - 2*dzpx) - dypy*dzpx + dxpy*(dypz - 2*dzpy) + dypx*dzpy + (2*dxypz - dxzpy - dyzpx)*py + (-dxypy + dyypx)*pz))+
                 Kt*((dyypz - dyzpy)*px*px + (dxxpz - dxzpx)*py*py + (-2*dxpy*dxpy + 4*dxpy*dypx - 2*dypx*dypx)*pz + py*(-dxpz*dypx + dxpy*(2*dxpz - 3*dzpx) + 2*dypx*dzpx + dxpx*(-dypz + dzpy) + (-dxxpy + dxypx)*pz) +
                     px*(-dxpz*dypy + dypy*dzpx + dypx*(2*dypz - 3*dzpy) + dxpy*(-dypz + 2*dzpy) + (-2*dxypz + dxzpy + dyzpx)*py + (dxypy - dyypx)*pz));


            k3[x] = h[x]/gama-nu*(Pol[x]*u[x][x]+Pol[y]*u[x][y]+Pol[z]*u[x][z]) + lambda*Pol[x]/delmu + (W[x][x]*Pol[x]+W[x][y]*Pol[y]+W[x][z]*Pol[z]);
            k3[y] = h[y]/gama-nu*(Pol[x]*u[y][x]+Pol[y]*u[y][y]+Pol[z]*u[y][z]) + lambda*Pol[y]/delmu + (W[y][x]*Pol[x]+W[y][y]*Pol[y]+W[y][z]*Pol[z]);
            k3[z] = h[z]/gama-nu*(Pol[x]*u[z][x]+Pol[y]*u[z][y]+Pol[z]*u[z][z]) + lambda*Pol[z]/delmu + (W[z][x]*Pol[x]+W[z][y]*Pol[y]+W[z][z]*Pol[z]);
            Pol_bulk = dPol + (dt) * k3;
            Particles.ghost_get<0>(SKIP_LABELLING);
            dxpx=Dx(Pol[x]);
            dxpy=Dx(Pol[y]);
            dxpz=Dx(Pol[z]);
            dypx=Dy(Pol[x]);
            dypy=Dy(Pol[y]);
            dypz=Dy(Pol[z]);
            dzpx=Dz(Pol[x]);
            dzpy=Dz(Pol[y]);
            dzpz=Dz(Pol[z]);
            dxxpx=Dxx(Pol[x]);
            dxxpy=Dxx(Pol[y]);
            dxxpz=Dxx(Pol[z]);
            dyypx=Dyy(Pol[x]);
            dyypy=Dyy(Pol[y]);
            dyypz=Dyy(Pol[z]);
            dzzpx=Dzz(Pol[x]);
            dzzpy=Dzz(Pol[y]);
            dzzpz=Dzz(Pol[z]);
            dxypx=Dxy(Pol[x]);
            dxypy=Dxy(Pol[y]);
            dxypz=Dxy(Pol[z]);
            dxzpx=Dxz(Pol[x]);
            dxzpy=Dxz(Pol[y]);
            dxzpz=Dxz(Pol[z]);
            dyzpx=Dyz(Pol[x]);
            dyzpy=Dyz(Pol[y]);
            dyzpz=Dyz(Pol[z]);
            h[x]=Ks*(dxxpx + dxypy + dxzpz) +
                 Kb*((-dxypy - dxzpz + dyypx + dzzpx)*px*px + (-dxypy + dyypx)*py*py + (dypy*dzpx + dxpy*(dypz - 2*dzpy) + dypx*dzpy + dxpz*(-dypy - 2*dzpz) + 2*dzpx*dzpz)*pz + (-dxzpz + dzzpx)*pz*pz +
                     py*(dypz*dzpx + dxpz*(-2*dypz + dzpy) + dxpy*(-2*dypy - dzpz) + dypx*(2*dypy + dzpz) + (-dxypz - dxzpy + 2*dyzpx)*pz) +
                     px*(-dxpy*dxpy - dxpz*dxpz + dypx*dypx + dypz*dypz + dzpx*dzpx - 2*dypz*dzpy + dzpy*dzpy + (-dyzpz + dzzpy)*py + (dyypz - dyzpy)*pz)) +
                 Kt*((-dxzpz + dzzpx)*py*py + (dxpz*dypy -  dypy*dzpx + dypx*(2*dypz -  dzpy) + dxpy*(-3*dypz + 2*dzpy))*pz + (- dxypy +  dyypx)*pz*pz + py*(-dypz*dzpx + dxpz*(2*dypz - 3*dzpy) + 2*dzpx*dzpy +  dxpy*dzpz -   dypx*dzpz + ( dxypz +  dxzpy - 2*dyzpx)*pz) +
                     px*(-2*dypz*dypz + 4*dypz*dzpy - 2*dzpy*dzpy + ( dyzpz -  dzzpy)*py + (- dyypz + dyzpy)*pz));

            h[y]= Ks*(dxypx + dyypy + dyzpz) +
                  Kb*((dxxpy - dxypx)*px*px + (dxxpy - dxypx - dyzpz + dzzpy)*py*py + (dxpz*dypx + dxpy*dzpx - 2*dypx*dzpx + dxpx*(-dypz + dzpy) - 2*dypz*dzpz + 2*dzpy*dzpz)*pz + (-dyzpz + dzzpy)*pz*pz +
                      py*(dxpy*dxpy + dxpz*dxpz - dypx*dypx - dypz*dypz - 2*dxpz*dzpx + dzpx*dzpx + dzpy*dzpy + (dxxpz - dxzpx)*pz) +
                      px*(dxpx*(2*dxpy - 2*dypx) + dypz*dzpx + dxpz*(-2*dypz + dzpy) + dxpy*dzpz - dypx*dzpz + (-dxzpz + dzzpx)*py + (-dxypz + 2*dxzpy - dyzpx)*pz)) +
                  Kt*((-dyzpz + dzzpy)*px*px + (-3*dxpz*dypx + dxpy*(2*dxpz - dzpx) + 2*dypx*dzpx + dxpx*(dypz - dzpy))*pz + (dxxpy - dxypx)*pz*pz + py*(-2*dxpz*dxpz + 4*dxpz*dzpx - 2*dzpx*dzpx + (-dxxpz + dxzpx)*pz) +
                      px*(-3*dypz*dzpx + dxpz*(2*dypz - dzpy) + 2*dzpx*dzpy - dxpy*dzpz + dypx*dzpz + (dxzpz - dzzpx)*py + (dxypz - 2*dxzpy + dyzpx)*pz));


            h[z]=Ks*(dxzpx + dyzpy + dzzpz) +
                 Kb*((dxxpz - dxzpx)*px*px + (dyypz - dyzpy)*py*py + (dxpy*dxpy + dxpz*dxpz - 2*dxpy*dypx + dypx*dypx + dypz*dypz - dzpx*dzpx - dzpy*dzpy)*pz + (dxxpz - dxzpx + dyypz - dyzpy)*pz*pz +
                     py*(dxpz*dypx + dxpy*dzpx - 2*dypx*dzpx + dypy*(2*dypz - 2*dzpy) + dxpx*(dypz - dzpy) + (dxxpy - dxypx)*pz) +
                     px*(dxpz*dypy + dxpx*(2*dxpz - 2*dzpx) - dypy*dzpx + dxpy*(dypz - 2*dzpy) + dypx*dzpy + (2*dxypz - dxzpy - dyzpx)*py + (-dxypy + dyypx)*pz))+
                 Kt*((dyypz - dyzpy)*px*px + (dxxpz - dxzpx)*py*py + (-2*dxpy*dxpy + 4*dxpy*dypx - 2*dypx*dypx)*pz + py*(-dxpz*dypx + dxpy*(2*dxpz - 3*dzpx) + 2*dypx*dzpx + dxpx*(-dypz + dzpy) + (-dxxpy + dxypx)*pz) +
                     px*(-dxpz*dypy + dypy*dzpx + dypx*(2*dypz - 3*dzpy) + dxpy*(-dypz + 2*dzpy) + (-2*dxypz + dxzpy + dyzpx)*py + (dxypy - dyypx)*pz));

            k4[x] = h[x]/gama-nu*(Pol[x]*u[x][x]+Pol[y]*u[x][y]+Pol[z]*u[x][z]) + lambda*Pol[x]/delmu + (W[x][x]*Pol[x]+W[x][y]*Pol[y]+W[x][z]*Pol[z]);
            k4[y] = h[y]/gama-nu*(Pol[x]*u[y][x]+Pol[y]*u[y][y]+Pol[z]*u[y][z]) + lambda*Pol[y]/delmu + (W[y][x]*Pol[x]+W[y][y]*Pol[y]+W[y][z]*Pol[z]);
            k4[z] = h[z]/gama-nu*(Pol[x]*u[z][x]+Pol[y]*u[z][y]+Pol[z]*u[z][z]) + lambda*Pol[z]/delmu + (W[z][x]*Pol[x]+W[z][y]*Pol[y]+W[z][z]*Pol[z]);

            //Pol_bulk = dPol + (dt / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4);
            /*for (int j = 0; j < Boundary.size(); j++) {
                auto p = Boundary.get<0>(j);
                Particles.getProp<0>(p)[x] = sin(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[y] = cos(2 * M_PI * (cos((2 * Particles.getPos(p)[x] - Lx) / Lx) -
                                                             sin((2 * Particles.getPos(p)[y] - Ly) / Ly)));
                Particles.getProp<0>(p)[z] =0;
            }*/

            k1 = V;
            k2 = 0.5 * dt * k1 + V;
            k3 = 0.5 * dt * k2 + V;
            k4 = dt * k3 + V;
            Pos = Pos + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);

            Particles.map();

            Particles.ghost_get<0, ExtForce>();
            indexUpdate(Particles,Boundary, bulk, up, down, left,right,front,back);
//            vector_dist_subset<3, double, aggregate<VectorS<3, double>, VectorS<3, double>, double[3][3], VectorS<3, double>, double, double[3][3], double[3][3], VectorS<3, double>, VectorS<3, double>, VectorS<3, double>, VectorS<3, double>, double, double, double, double, double, double, double, VectorS<3, double>, double, double, double[3], double[3], double[3], double[3], double[3], double[3], double, VectorS<3, double>, VectorS<3, double>, VectorS<3, double>, VectorS<3, double>, double, double, double, double[3][3],double[3][3]>> Particles_subset(
//                    Particles, bulk);

            Particles_subset.update(bulk);
            auto Pol_bulk = getV<0>(Particles_subset);
            auto P_bulk = getV<Pressure>(Particles_subset);
            auto dPol_bulk = getV<8>(Particles_subset);
            auto RHS_bulk = getV<10>(Particles_subset);



            //Particles_subset.write("debug");

            tt.start();
            Dx.update(Particles);
            Dy.update(Particles);
            Dz.update(Particles);
            Dxy.update(Particles);
            Dxz.update(Particles);
            Dyz.update(Particles);
            auto Dyx = Dxy;
            auto Dzy = Dyz;
            auto Dzx = Dxz;
            Dxx.update(Particles);
            Dyy.update(Particles);
            Dzz.update(Particles);

            Bulk_Dx.update(Particles_subset);
            Bulk_Dy.update(Particles_subset);
            Bulk_Dz.update(Particles_subset);


            tt.stop();
            if (v_cl.rank() == 0) {
                std::cout << "Updation of operators took " << tt.getwct() << " seconds." << std::endl;
                std::cout << "Time step " << ctr - 1 << " : " << tim << " over." << std::endl;
                std::cout << "----------------------------------------------------------" << std::endl;
            }

            tim += dt;
            Particles.ghost_get<Polarization>(SKIP_LABELLING);
            dxpx=Dx(Pol[x]);
            dxpy=Dx(Pol[y]);
            dxpz=Dx(Pol[z]);
            dypx=Dy(Pol[x]);
            dypy=Dy(Pol[y]);
            dypz=Dy(Pol[z]);
            dzpx=Dz(Pol[x]);
            dzpy=Dz(Pol[y]);
            dzpz=Dz(Pol[z]);
            dxxpx=Dxx(Pol[x]);
            dxxpy=Dxx(Pol[y]);
            dxxpz=Dxx(Pol[z]);
            dyypx=Dyy(Pol[x]);
            dyypy=Dyy(Pol[y]);
            dyypz=Dyy(Pol[z]);
            dzzpx=Dzz(Pol[x]);
            dzzpy=Dzz(Pol[y]);
            dzzpz=Dzz(Pol[z]);
            dxypx=Dxy(Pol[x]);
            dxypy=Dxy(Pol[y]);
            dxypz=Dxy(Pol[z]);
            dxzpx=Dxz(Pol[x]);
            dxzpy=Dxz(Pol[y]);
            dxzpz=Dxz(Pol[z]);
            dyzpx=Dyz(Pol[x]);
            dyzpy=Dyz(Pol[y]);
            dyzpz=Dyz(Pol[z]);
            dxqxx=Dx(Pol[x]*Pol[x]-1/3*(Pol[x]*Pol[x]+Pol[y]*Pol[y]+Pol[z]*Pol[z]));
            dyqxy=Dy(Pol[x]*Pol[x]);
            dzqxz=Dz(Pol[x]*Pol[z]);
            dxqyx=Dx(Pol[y]*Pol[x]);
            dyqyy=Dy(Pol[y]*Pol[y]-1/3*(Pol[x]*Pol[x]+Pol[y]*Pol[y]+Pol[z]*Pol[z]));
            dzqyz=Dz(Pol[y]*Pol[z]);
            dxqzx=Dx(Pol[z]*Pol[x]);
            dyqzy=Dy(Pol[z]*Pol[y]);
            dzqzz=Dz(Pol[z]*Pol[z]-1/3*(Pol[x]*Pol[x]+Pol[y]*Pol[y]+Pol[z]*Pol[z]));

        }

        Particles.deleteGhost();
        Particles.write("Polar_Last");

        Dx.deallocate(Particles);
        Dy.deallocate(Particles);
        Dz.deallocate(Particles);
        Dxy.deallocate(Particles);
        Dxz.deallocate(Particles);
        Dyz.deallocate(Particles);
        Dxx.deallocate(Particles);
        Dyy.deallocate(Particles);
        Dzz.deallocate(Particles);
        Bulk_Dx.deallocate(Particles_subset);
        Bulk_Dy.deallocate(Particles_subset);
        Bulk_Dz.deallocate(Particles_subset);
        Particles.deleteGhost();
        tt2.stop();
        if (v_cl.rank() == 0) {
            std::cout << "The simulation took " << tt2.getcputime() << "(CPU) ------ " << tt2.getwct()
                      << "(Wall) Seconds.";
        }
    }



BOOST_AUTO_TEST_SUITE_END()
