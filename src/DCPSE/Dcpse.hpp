//
// Created by tommaso on 29/03/19.
// Modified by Abhinav and Pietro
#ifndef OPENFPM_PDATA_DCPSE_HPP
#define OPENFPM_PDATA_DCPSE_HPP

#ifdef HAVE_EIGEN

#include "Vector/vector_dist.hpp"
#include "MonomialBasis.hpp"
#include "DMatrix/EMatrix.hpp"
#include "SupportBuilder.hpp"
#include "Support.hpp"
#include "Vandermonde.hpp"
#include "DcpseDiagonalScalingMatrix.hpp"
#include "DcpseRhs.hpp"

template<bool cond>
struct is_scalar {
	template<typename op_type>
	static auto
	analyze(const vect_dist_key_dx &key, op_type &o1) -> typename std::remove_reference<decltype(o1.value(
			key))>::type {
		return o1.value(key);
	};
};

template<>
struct is_scalar<false> {
	template<typename op_type>
	static auto
	analyze(const vect_dist_key_dx &key, op_type &o1) -> typename std::remove_reference<decltype(o1.value(
			key))>::type {
		return o1.value(key);
	};
};

template<unsigned int dim, typename vector_type>
class Dcpse {
public:

    typedef typename vector_type::stype T;
    typedef typename vector_type::value_type part_type;
    typedef vector_type vtype;

    #ifdef SE_CLASS1
    int update_ctr=0;
    #endif

    // This works in this way:
    // 1) User constructs this by giving a domain of points (where one of the properties is the value of our f),
    //    the signature of the differential operator and the error order bound.
    // 2) The machinery for assembling and solving the linear system for coefficients starts...
    // 3) The user can then call an evaluate(point) method to get the evaluation of the differential operator
    //    on the given point.
private:
    const Point<dim, unsigned int> differentialSignature;
    const unsigned int differentialOrder;
    const MonomialBasis<dim> monomialBasis;
    //std::vector<EMatrix<T, Eigen::Dynamic, 1>> localCoefficients; // Each MPI rank has just access to the local ones
    std::vector<Support> localSupports; // Each MPI rank has just access to the local ones
    std::vector<T> localEps; // Each MPI rank has just access to the local ones
    std::vector<T> localEpsInvPow; // Each MPI rank has just access to the local ones
    std::vector<T> localSumA;

    openfpm::vector<size_t> kerOffsets;
    openfpm::vector<T> calcKernels;

    vector_type & particles;
    double rCut;
    unsigned int convergenceOrder;
    double supportSizeFactor;

    support_options opt;

public:
#ifdef SE_CLASS1
    int getUpdateCtr() const
    {
        return update_ctr;
    }
#endif

    // Here we require the first element of the aggregate to be:
    // 1) the value of the function f on the point
    Dcpse(vector_type &particles,
          Point<dim, unsigned int> differentialSignature,
          unsigned int convergenceOrder,
          T rCut,
          T supportSizeFactor = 1,                               //Maybe change this to epsilon/h or h/epsilon = c 0.9. Benchmark
          support_options opt = support_options::RADIUS)
		:particles(particles),
            differentialSignature(differentialSignature),
            differentialOrder(Monomial<dim>(differentialSignature).order()),
            monomialBasis(differentialSignature.asArray(), convergenceOrder),
            opt(opt)
    {
        // This 
        particles.ghost_get_subset();
        if (supportSizeFactor < 1) 
        {
            initializeAdaptive(particles, convergenceOrder, rCut);
        } 
        else 
        {
            initializeStaticSize(particles, convergenceOrder, rCut, supportSizeFactor);
        }
    }



    template<unsigned int prp>
    void DrawKernel(vector_type &particles, int k)
    {
        Support support = localSupports[k];
        size_t xpK = k;
        size_t kerOff = kerOffsets.get(k);
        auto & keys = support.getKeys();
        for (int i = 0 ; i < keys.size() ; i++)
        {
        	size_t xqK = keys[i];
            particles.template getProp<prp>(xqK) += calcKernels.get(kerOff+i);
        }
    }

    template<unsigned int prp>
    void DrawKernelNN(vector_type &particles, int k)
    {
        Support support = localSupports[k];
        size_t xpK = k;
        size_t kerOff = kerOffsets.get(k);
        auto & keys = support.getKeys();
        for (int i = 0 ; i < keys.size() ; i++)
        {
        	size_t xqK = keys[i];
            particles.template getProp<prp>(xqK) = 1.0;
        }
    }

    template<unsigned int prp>
    void DrawKernel(vector_type &particles, int k, int i)
    {
        Support support = localSupports[k];
        size_t xpK = k;
        size_t kerOff = kerOffsets.get(k);
        auto & keys = support.getKeys();
        for (int i = 0 ; i < keys.size() ; i++)
        {
        	size_t xqK = keys[i];
            particles.template getProp<prp>(xqK)[i] += calcKernels.get(kerOff+i);
        }
    }

    void checkMomenta(vector_type &particles)
    {
        openfpm::vector<aggregate<double,double>> momenta;
        openfpm::vector<aggregate<double,double>> momenta_accu;

        momenta.resize(monomialBasis.size());
        momenta_accu.resize(monomialBasis.size());

        for (int i = 0 ; i < momenta.size() ; i++)
        {
            momenta.template get<0>(i) =  3000000000.0;
            momenta.template get<1>(i) = -3000000000.0;
        }

        auto it = particles.getDomainIterator();
        auto supportsIt = localSupports.begin();
        auto epsIt = localEps.begin();
        while (it.isNext())
        {
            double eps = *epsIt;

            for (int i = 0 ; i < momenta.size() ; i++)
            {
                momenta_accu.template get<0>(i) =  0.0;
            }

            Support support = *supportsIt;
            size_t xpK = support.getReferencePointKey();
            Point<dim, T> xp = particles.getPos(support.getReferencePointKey());
            size_t kerOff = kerOffsets.get(xpK);
            auto & keys = support.getKeys();
            for (int i = 0 ; i < keys.size() ; i++)
            {
            	size_t xqK = keys[i];
                Point<dim, T> xq = particles.getPosOrig(xqK);
                Point<dim, T> normalizedArg = (xp - xq) / eps;

                auto ker = calcKernels.get(kerOff+i);

                int counter = 0;
                for (const Monomial<dim> &m : monomialBasis.getElements())
                {
                    T mbValue = m.evaluate(normalizedArg);


                    momenta_accu.template get<0>(counter) += mbValue * ker;

                    ++counter;
                }

            }

            for (int i = 0 ; i < momenta.size() ; i++)
            {
                if (momenta_accu.template get<0>(i) < momenta.template get<0>(i))
                {
                    momenta.template get<0>(i) = momenta_accu.template get<0>(i);
                }

                if (momenta_accu.template get<1>(i) > momenta.template get<1>(i))
                {
                    momenta.template get<1>(i) = momenta_accu.template get<0>(i);
                }
            }

            //
            ++it;
            ++supportsIt;
            ++epsIt;
        }

        for (int i = 0 ; i < momenta.size() ; i++)
        {
            std::cout << "MOMENTA: " << monomialBasis.getElements()[i] << "Min: " << momenta.template get<0>(i) << "  " << "Max: " << momenta.template get<1>(i) << std::endl;
        }
    }

    /**
     * Computes the value of the differential operator on all the particles,
     * using the f values stored at the fValuePos position in the aggregate
     * and storing the resulting Df values at the DfValuePos position in the aggregate.
     * @tparam fValuePos Position in the aggregate of the f values to use.
     * @tparam DfValuePos Position in the aggregate of the Df values to store.
     * @param particles The set of particles to iterate over.
     */
    template<unsigned int fValuePos, unsigned int DfValuePos>
    void computeDifferentialOperator(vector_type &particles) {
        char sign = 1;
        if (differentialOrder % 2 == 0) {
            sign = -1;
        }

        auto it = particles.getDomainIterator();
        auto supportsIt = localSupports.begin();
        auto epsItInvPow = localEpsInvPow.begin();
        while (it.isNext()) {
            double epsInvPow = *epsItInvPow;

            T Dfxp = 0;
            Support support = *supportsIt;
            size_t xpK = support.getReferencePointKey();
            Point<dim, typename vector_type::stype> xp = particles.getPos(support.getReferencePointKey());
            T fxp = sign * particles.template getProp<fValuePos>(xpK);
            size_t kerOff = kerOffsets.get(xpK);
            auto & keys = support.getKeys();
            for (int i = 0 ; i < keys.size() ; i++)
            {
            	size_t xqK = keys[i];
                T fxq = particles.template getProp<fValuePos>(xqK);

                Dfxp += (fxq + fxp) * calcKernels.get(kerOff+i);
            }
            Dfxp *= epsInvPow;
            //
            //T trueDfxp = particles.template getProp<2>(xpK);
            // Store Dfxp in the right position
            particles.template getProp<DfValuePos>(xpK) = Dfxp;
            //
            ++it;
            ++supportsIt;
            ++epsItInvPow;
        }
    }


    /*! \brief Get the number of neighbours
     *
     * \return the number of neighbours
     *
     */
    inline int getNumNN(const vect_dist_key_dx &key)
    {
        return localSupports[key.getKey()].size();
    }

    /*! \brief Get the coefficent j (Neighbour) of the particle key
     *
     * \param key particle
     * \param j neighbour
     *
     * \return the coefficent
     *
     */
    inline T getCoeffNN(const vect_dist_key_dx &key, int j)
    {
    	size_t base = kerOffsets.get(key.getKey());
        return calcKernels.get(base + j);
    }

    /*! \brief Get the number of neighbours
 *
 * \return the number of neighbours
 *
 */
    inline size_t getIndexNN(const vect_dist_key_dx &key, int j)
    {
        return localSupports[key.getKey()].getKeys()[j];
    }


    inline T getSign()
    {
        T sign = 1.0;
        if (differentialOrder % 2 == 0) {
            sign = -1;
        }

        return sign;
    }

    T getEpsilonInvPrefactor(const vect_dist_key_dx &key)
    {
        return localEpsInvPow[key.getKey()];
    }

    /**
     * Computes the value of the differential operator for one particle for o1 representing a scalar
     *
     * \param key particle
     * \param o1 source property
     * \return the selected derivative
     *
     */
    template<typename op_type>
    auto computeDifferentialOperator(const vect_dist_key_dx &key,
                                     op_type &o1) -> decltype(is_scalar<std::is_fundamental<decltype(o1.value(
            key))>::value>::analyze(key, o1)) {

        typedef decltype(is_scalar<std::is_fundamental<decltype(o1.value(key))>::value>::analyze(key, o1)) expr_type;

        T sign = 1.0;
        if (differentialOrder % 2 == 0) {
            sign = -1;
        }

        double eps = localEps[key.getKey()];
        double epsInvPow = localEpsInvPow[key.getKey()];

        auto &particles = o1.getVector();

#ifdef SE_CLASS1
        if(particles.getMapCtr()!=this->getUpdateCtr())
        {
            std::cerr<<__FILE__<<":"<<__LINE__<<" Error: You forgot a DCPSE operator update after map."<<std::endl;
        }
#endif

        expr_type Dfxp = 0;
        Support support = localSupports[key.getKey()];
        size_t xpK = support.getReferencePointKey();
        Point<dim, T> xp = particles.getPos(xpK);
        expr_type fxp = sign * o1.value(key);
        size_t kerOff = kerOffsets.get(xpK);
        auto & keys = support.getKeys();
        for (int i = 0 ; i < keys.size() ; i++)
        {
        	size_t xqK = keys[i];
            expr_type fxq = o1.value(vect_dist_key_dx(xqK));
            Dfxp = Dfxp + (fxq + fxp) * calcKernels.get(kerOff+i);
        }
        Dfxp = Dfxp * epsInvPow;
        //
        //T trueDfxp = particles.template getProp<2>(xpK);
        // Store Dfxp in the right position
        return Dfxp;
    }

    /**
     * Computes the value of the differential operator for one particle for o1 representing a vector
     *
     * \param key particle
     * \param o1 source property
     * \param i component
     * \return the selected derivative
     *
     */
    template<typename op_type>
    auto computeDifferentialOperator(const vect_dist_key_dx &key,
                                     op_type &o1,
                                     int i) -> typename decltype(is_scalar<std::is_fundamental<decltype(o1.value(
            key))>::value>::analyze(key, o1))::coord_type {

        typedef typename decltype(is_scalar<std::is_fundamental<decltype(o1.value(key))>::value>::analyze(key, o1))::coord_type expr_type;

        //typedef typename decltype(o1.value(key))::blabla blabla;

        T sign = 1.0;
        if (differentialOrder % 2 == 0) {
            sign = -1;
        }

        double eps = localEps[key.getKey()];
        double epsInvPow = localEpsInvPow[key.getKey()];

        auto &particles = o1.getVector();

#ifdef SE_CLASS1
        if(particles.getMapCtr()!=this->getUpdateCtr())
        {
            std::cerr<<__FILE__<<":"<<__LINE__<<" Error: You forgot a DCPSE operator update after map."<<std::endl;
        }
#endif

        expr_type Dfxp = 0;
        Support support = localSupports[key.getKey()];
        size_t xpK = support.getReferencePointKey();
        Point<dim, T> xp = particles.getPos(xpK);
        expr_type fxp = sign * o1.value(key)[i];
        size_t kerOff = kerOffsets.get(xpK);
        auto & keys = support.getKeys();
        for (int j = 0 ; j < keys.size() ; j++)
        {
        	size_t xqK = keys[j];
            expr_type fxq = o1.value(vect_dist_key_dx(xqK))[i];
            Dfxp = Dfxp + (fxq + fxp) * calcKernels.get(kerOff+j);
        }
        Dfxp = Dfxp * epsInvPow;
        //
        //T trueDfxp = particles.template getProp<2>(xpK);
        // Store Dfxp in the right position
        return Dfxp;
    }

    void initializeUpdate(vector_type &particles)
    {
#ifdef SE_CLASS1
        update_ctr=particles.getMapCtr();
#endif

        localSupports.clear();
        localSupports.resize(particles.size_local_orig());
        localEps.clear();
        localEps.resize(particles.size_local_orig());
        localEpsInvPow.clear();
        localEpsInvPow.resize(particles.size_local_orig());
        calcKernels.clear();
        kerOffsets.clear();
        kerOffsets.resize(particles.size_local_orig());
        kerOffsets.fill(-1);

        SupportBuilder<vector_type> supportBuilder(particles, differentialSignature, rCut);
        unsigned int requiredSupportSize = monomialBasis.size() * supportSizeFactor;

        auto it = particles.getDomainIterator();
        while (it.isNext()) {
            // Get the points in the support of the DCPSE kernel and store the support for reuse
            //Support<vector_type> support = supportBuilder.getSupport(it, requiredSupportSize,opt);
            Support support = supportBuilder.getSupport(it, requiredSupportSize,opt);
            EMatrix<T, Eigen::Dynamic, Eigen::Dynamic> V(support.size(), monomialBasis.size());

            auto key_o = particles.getOriginKey(it.get());

            // Vandermonde matrix computation
            Vandermonde<dim, T, EMatrix<T, Eigen::Dynamic, Eigen::Dynamic>>
                    vandermonde(support, monomialBasis,particles);
            vandermonde.getMatrix(V);

            T eps = vandermonde.getEps();

            localSupports[key_o.getKey()] = support;
            localEps[key_o.getKey()] = eps;
            localEpsInvPow[key_o.getKey()] = 1.0 / openfpm::math::intpowlog(eps,differentialOrder);
            // Compute the diagonal matrix E
            DcpseDiagonalScalingMatrix<dim> diagonalScalingMatrix(monomialBasis);
            EMatrix<T, Eigen::Dynamic, Eigen::Dynamic> E(support.size(), support.size());
            diagonalScalingMatrix.buildMatrix(E, support, eps, particles);
            // Compute intermediate matrix B
            EMatrix<T, Eigen::Dynamic, Eigen::Dynamic> B = E * V;
            // Compute matrix A
            EMatrix<T, Eigen::Dynamic, Eigen::Dynamic> A = B.transpose() * B;

            // Compute RHS vector b
            DcpseRhs<dim> rhs(monomialBasis, differentialSignature);
            EMatrix<T, Eigen::Dynamic, 1> b(monomialBasis.size(), 1);
            rhs.template getVector<T>(b);
            // Get the vector where to store the coefficients...
            EMatrix<T, Eigen::Dynamic, 1> a(monomialBasis.size(), 1);
            // ...solve the linear system...
            a = A.colPivHouseholderQr().solve(b);
            // ...and store the solution for later reuse
            kerOffsets.get(key_o.getKey()) = calcKernels.size();

            Point<dim, T> xp = particles.getPosOrig(key_o);

            for (auto &xqK : support.getKeys())
            {
                Point<dim, T> xq = particles.getPosOrig(xqK);
                Point<dim, T> normalizedArg = (xp - xq) / eps;

                calcKernels.add(computeKernel(normalizedArg, a));
            }
            //
            ++it;
        }
    }

private:

    void initializeAdaptive(vector_type &particles,
                            unsigned int convergenceOrder,
                            T rCut) {
        SupportBuilder<vector_type>
                supportBuilder(particles, differentialSignature, rCut);
        unsigned int requiredSupportSize = monomialBasis.size();

        localSupports.resize(particles.size_local_orig());
        localEps.resize(particles.size_local_orig());
        localEpsInvPow.resize(particles.size_local_orig());
        kerOffsets.resize(particles.size_local_orig());
        kerOffsets.fill(-1);

        auto it = particles.getDomainIterator();
        while (it.isNext()) {
            const T condVTOL = 1e2;

            // Get the points in the support of the DCPSE kernel and store the support for reuse
            Support support = supportBuilder.getSupport(it, requiredSupportSize,opt);
            EMatrix<T, Eigen::Dynamic, Eigen::Dynamic> V(support.size(), monomialBasis.size());

            // Vandermonde matrix computation
            Vandermonde<dim, T, EMatrix<T, Eigen::Dynamic, Eigen::Dynamic>>
                    vandermonde(support, monomialBasis, particles);
            vandermonde.getMatrix(V);

            T condV = conditionNumber(V, condVTOL);
            T eps = vandermonde.getEps();

            if (condV > condVTOL) {
                requiredSupportSize *= 2;
                std::cout
                        << "INFO: Increasing, requiredSupportSize = " << requiredSupportSize
                        << std::endl; // debug
                continue;
            } else {
                requiredSupportSize = monomialBasis.size();
            }

            auto key_o = particles.getOriginKey(it.get());
            localSupports[key_o.getKey()] = support;
            localEps[key_o.getKey()] = eps;
            localEpsInvPow[key_o.getKey()] = 1.0 / openfpm::math::intpowlog(eps,differentialOrder);
            // Compute the diagonal matrix E
            DcpseDiagonalScalingMatrix<dim> diagonalScalingMatrix(monomialBasis);
            EMatrix<T, Eigen::Dynamic, Eigen::Dynamic> E(support.size(), support.size());
            diagonalScalingMatrix.buildMatrix(E, support, eps,particles);
            // Compute intermediate matrix B
            EMatrix<T, Eigen::Dynamic, Eigen::Dynamic> B = E * V;
            // Compute matrix A
            EMatrix<T, Eigen::Dynamic, Eigen::Dynamic> A = B.transpose() * B;
            // Compute RHS vector b
            DcpseRhs<dim> rhs(monomialBasis, differentialSignature);
            EMatrix<T, Eigen::Dynamic, 1> b(monomialBasis.size(), 1);
            rhs.template getVector<T>(b);
            // Get the vector where to store the coefficients...
            EMatrix<T, Eigen::Dynamic, 1> a(monomialBasis.size(), 1);
            // ...solve the linear system...
            a = A.colPivHouseholderQr().solve(b);
            // ...and store the solution for later reuse
            kerOffsets.get(key_o.getKey()) = calcKernels.size();

            Point<dim, T> xp = particles.getPosOrig(key_o);

            for (auto &xqK : support.getKeys())
            {
                Point<dim, T> xq = particles.getPosOrig(xqK);
                Point<dim, T> normalizedArg = (xp - xq) / eps;

                calcKernels.add(computeKernel(normalizedArg, a));
            }
            //
            ++it;
        }
    }


    void initializeStaticSize(vector_type &particles,
                              unsigned int convergenceOrder,
                              T rCut,
                              T supportSizeFactor) {
#ifdef SE_CLASS1
        this->update_ctr=particles.getMapCtr();
#endif
        this->rCut=rCut;
        this->supportSizeFactor=supportSizeFactor;
        this->convergenceOrder=convergenceOrder;
        SupportBuilder<vector_type>
                supportBuilder(particles, differentialSignature, rCut);
        unsigned int requiredSupportSize = monomialBasis.size() * supportSizeFactor;

        localSupports.resize(particles.size_local_orig());
        localEps.resize(particles.size_local_orig());
        localEpsInvPow.resize(particles.size_local_orig());
        kerOffsets.resize(particles.size_local_orig());

        auto it = particles.getDomainIterator();
        while (it.isNext()) {
            // Get the points in the support of the DCPSE kernel and store the support for reuse
            Support support = supportBuilder.getSupport(it, requiredSupportSize,opt);
            EMatrix<T, Eigen::Dynamic, Eigen::Dynamic> V(support.size(), monomialBasis.size());

            auto key_o = particles.getOriginKey(it.get());

            // Vandermonde matrix computation
            Vandermonde<dim, T, EMatrix<T, Eigen::Dynamic, Eigen::Dynamic>>
                    vandermonde(support, monomialBasis,particles);
            vandermonde.getMatrix(V);

            T eps = vandermonde.getEps();

            localSupports[key_o.getKey()] = support;
            localEps[key_o.getKey()] = eps;
            localEpsInvPow[key_o.getKey()] = 1.0 / openfpm::math::intpowlog(eps,differentialOrder);
            // Compute the diagonal matrix E
            DcpseDiagonalScalingMatrix<dim> diagonalScalingMatrix(monomialBasis);
            EMatrix<T, Eigen::Dynamic, Eigen::Dynamic> E(support.size(), support.size());
            diagonalScalingMatrix.buildMatrix(E, support, eps, particles);
            // Compute intermediate matrix B
            EMatrix<T, Eigen::Dynamic, Eigen::Dynamic> B = E * V;
            // Compute matrix A
            EMatrix<T, Eigen::Dynamic, Eigen::Dynamic> A = B.transpose() * B;

            // Compute RHS vector b
            DcpseRhs<dim> rhs(monomialBasis, differentialSignature);
            EMatrix<T, Eigen::Dynamic, 1> b(monomialBasis.size(), 1);
            rhs.template getVector<T>(b);
            // Get the vector where to store the coefficients...
            EMatrix<T, Eigen::Dynamic, 1> a(monomialBasis.size(), 1);
            // ...solve the linear system...
            a = A.colPivHouseholderQr().solve(b);
            // ...and store the solution for later reuse
            kerOffsets.get(key_o.getKey()) = calcKernels.size();

            Point<dim, T> xp = particles.getPosOrig(key_o);

            for (auto &xqK : support.getKeys())
            {
                Point<dim, T> xq = particles.getPosOrig(xqK);
                Point<dim, T> normalizedArg = (xp - xq) / eps;

                calcKernels.add(computeKernel(normalizedArg, a));
            }
            //
            ++it;
        }
    }




    T computeKernel(Point<dim, T> x, EMatrix<T, Eigen::Dynamic, 1> & a) const {
        T res = 0;
        unsigned int counter = 0;
        T expFactor = exp(-norm2(x));
        for (const Monomial<dim> &m : monomialBasis.getElements()) {
            T coeff = a(counter);
            T mbValue = m.evaluate(x);
            res += coeff * mbValue * expFactor;
            ++counter;
        }
        return res;
    }


    T conditionNumber(const EMatrix<T, -1, -1> &V, T condTOL) const {
        Eigen::JacobiSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd(V);
        T cond = svd.singularValues()(0)
                 / svd.singularValues()(svd.singularValues().size() - 1);
        if (cond > condTOL) {
            std::cout
                    << "WARNING: cond(V) = " << cond
                    << " is greater than TOL = " << condTOL
                    << ",  numPoints(V) = " << V.rows()
                    << std::endl; // debug
        }
        return cond;
    }

};


#endif
#endif //OPENFPM_PDATA_DCPSE_HPP

