from numpy import argmax
import itertools

def order_denominator(O):
    B = O.quaternion_algebra()
    O0 = B.maximal_order()
    I = connecting_ideal(O0, O)
    J = small_equivalent_ideal(I)
    return J.norm()

def trace_zero_norm_form(I, scaled_to_integal=False):
    Q = I.quaternion_algebra()
    a, b = Q.invariants()
    basis = lower_hnf_basis(Q, I.basis())
    R = PolynomialRing(Rationals(), names=["x1", "x2", "x3"])
    x1, x2, x3 = R.gens()
    form = (basis[1][0]*x1 + basis[2][0]*x2  + basis[3][0]*x3)**2 - a*(basis[1][1]*x1 + basis[2][1]*x2  + basis[3][1]*x3)**2 - b*(basis[1][2]*x1 + basis[2][2]*x2  + basis[3][2]*x3)**2 + a*b*(basis[1][3]*x1 + basis[2][3]*x2 + basis[3][3]*x3)**2
    if not scaled_to_integal:
        return QuadraticForm(form)
    M = QuadraticForm(form).matrix()
    N = lcm([denominator(a) for a in M.list()])
    M = M * N
    return QuadraticForm(M)

def enum_cyclic_l_ideals(O, ell):
    yield O.unit_ideal()
    n = 0
    while True:
        n += 1
        for I in left_ideals_of_norm_enumerator(O, ell**n):
            if is_cyclic(I) == False:
                continue
            yield I

def is_power_of_ell(N, ell):
    logN = log(N, ell)
    return ceil(logN) == floor(logN)

def make_ell_cyclic(I, ell):
    O = I.left_order()
    Ibasis = I.basis()
    while True:
        new_basis = [ei*(1/ell) for ei in Ibasis]
        if False in [ei in O for ei in new_basis]:
            return O.left_ideal(Ibasis)
        Ibasis = new_basis

def quadratic_order_from_norm_trace(d, t, generator=False):
    """Returns quadratic order where the generator has given norm and trace"""
    disc = t**2 - 4*d  
    K = QuadraticField(disc)
    z = K.gens()[0]
    g = (t + z)/2
    if generator:
        return K.order([1, g]), g
    return K.order([1, g])

def connecting_ideal(O0, O1):
    I = O0 * O1
    I = I * denominator(I.norm())
    return I

def small_equivalent_ideal(I, certificate=False):
    """
    Returns a left-ideal J of smaller norm in the right ideal class.
    If `certificate` is True, also returns the element y such that Iy = J
    """
    _,mn = I.quadratic_form().__pari__().qfminim(None,None,1)
    el = sum(ZZ(c)*g for c,g in zip(mn, I.basis()))
    y = el.conjugate() / I.norm()
    I *= y
    if certificate:
        return I, y
    return I

def Fp_to_int(n):
    """
    Returns element of GF(p) as integer in range (-p/2 ... +p/2]
    """
    if ZZ(n) > n.parent().order() / 2: return ZZ(n) - n.parent().order()
    return ZZ(n)

def is_cyclic(I, giveGcd=False):
    """
    Computes whether the input ideal is cyclic,
    all the work is done by the helper function
    `ideal_basis_gcd()`.
    From https://github.com/LearningToSQI/SQISign-SageMath/blob/main/ideals.py
    """

    def ideal_basis_gcd(I):
        """
        Computes the gcd of the coefficients of
        the ideal written as a linear combination
        of the basis of its left order.
        From https://github.com/LearningToSQI/SQISign-SageMath/blob/main/ideals.py
        """
        I_basis = I.basis_matrix()
        O_basis = I.left_order().unit_ideal().basis_matrix()

        # Write I in the basis of its left order
        M = I_basis * O_basis.inverse()
        return gcd((gcd(M_row) for M_row in M))

    g = ideal_basis_gcd(I)
    if giveGcd:
        return g == 1, g
    return g == 1

def order_with_denominator_coprime(O, ell, certificate=False):
    """
    Returns quaternion order isomorphic to `O` with denominator coprime to `ell`.
    If certificate is true returns the element to conjugate by also.
    From https://github.com/sagemath/sage/pull/37103
    """
    B = O.quaternion_algebra()

    if O.basis_matrix().denominator() % ell != 0:
        return O, B.one()

    O0 = B.maximal_order()  # <- The order with denominator 2q
    I_conn = O0 * O
    I_conn *= 1 / I_conn.norm()

    while True:
        alpha = sum([ZZ.random_element(-100, 100)*beta for beta in I_conn.basis()])
        if gcd(alpha.reduced_norm(), I_conn.norm()**2) == I_conn.norm():
            break

    alpha /= I_conn.norm()
    O_new = B.quaternion_order([alpha * g * ~alpha for g in O.basis()])
    if certificate:
        return O_new, alpha
    return O_new

def _P1_to_ideals(O, ell):
    """
    From https://github.com/sagemath/sage/pull/37103
    """
    B = O.quaternion_algebra()
    if ell in B.ramified_primes():
        raise ValueError(f"ell(={ell}) is not split")

    if not O.quadratic_form().is_positive_definite():
        raise TypeError("the quaternion algebra must be positive definite")

    phi = B.modp_splitting_map(ell)
    F = GF(ell)

    O_new, gamma = order_with_denominator_coprime(O, ell, True)
    mat_basis = [phi(beta) for beta in O_new.basis()]
    system = matrix(F, [[mat_basis[j][i // 2][i % 2] for j in range(4)] for i in range(4)])

    def return_map(a, b):
        a = a % ell
        b = b % ell
        assert not (a == b == 0)

        x, y, z, w = [ZZ(n) for n in system.solve_right(vector(F, [a, b, 0, 0]))]
        alpha = sum(c*beta for (c, beta) in zip([x,y,z,w], O_new.basis()))
        # assert (O_new*alpha + O_new*ell).norm() == ell
        return ~gamma * (O_new*alpha + O_new*ell) * gamma

    return return_map

def random_left_ideal_of_norm(O, N, primitive=True):
    """
    Given an order `O` and integer `N`, returns a left ideal of norm `N`.
    From https://github.com/sagemath/sage/pull/37103
    """

    def _random_left_ideal_2q(O, N):
        alpha = O.random_element()
        while gcd(alpha.reduced_norm(), N*N) != N:
            alpha = O.random_element()
        return O*alpha + O*N

    if O.base_ring() != ZZ:
        raise NotImplementedError("only implemented for quaternion algebras over ZZ")

    # it seems this check is not strict, as the method still works sometimes
    if gcd(O.discriminant(), N) != 1:
        raise ValueError("the norm must be coprime with the discriminant")

    B, (i, j, _) = O.quaternion_algebra().objgens()

    # the method is not properly implemented when B is not a division algebra
    # TODO: remove this warning
    if not B.is_division_algebra():
        import warnings
        warnings.warn("B is not a division algebra. The method may run into an infinite loop.")

    I = O.unit_ideal()
    I_last = I
    O_i = O

    for (ell, e) in factor(N):
        cnt = 0
        # repeat until `e` successful iterations
        while cnt < e:
            if gcd(ell, 2 * ZZ(i**2 * j**2)) != 1:
                while True:
                    I_ab = _random_left_ideal_2q(O_i, ell)
                    if not primitive or I_ab.conjugate() != I_last:
                        break
            else:
                mapping = _P1_to_ideals(O_i, ell)
                while True:
                    # Random element of P1(GF(ell))
                    x = ZZ.random_element(ell + 1)
                    a, b = 1 - x // ell, x + 1
                    I_ab = mapping(a, b)
                    if not primitive or I_ab.conjugate() != I_last:
                        break

            try:
                O_i = I_ab.right_order()
                I = I * I_ab
                I_last = I_ab
                cnt += 1
            except ZeroDivisionError:
                pass

    return I

def random_ideal(O):
    """
    Returns a random left-ideal of quaternion maximal order `O`.
    Works by sampling a random ideal of norm > p^2, then finding the smallest equivalent ideal.
    """
    B = O.quaternion_algebra()
    pmax = max([-p for p in B.invariants()])
    pbigger = ZZ(randrange(ZZ(pmax**2), ZZ(pmax**2)*2 + 2000)).next_prime()
    return small_equivalent_ideal(random_left_ideal_of_norm(O, pbigger))

def random_maximal_order(B):
    """Returns a random maximal order in the quaternion algebra B"""
    O0 = B.maximal_order()
    return random_ideal(O0).right_order()

def flattened_factors(N):
    """Given integer N, returns array of factors where prime factors appear multiple times depending on exponent. E.g. 12 gives [2, 2, 3]"""
    Nfact = factor(N)
    arr = []
    for fact in Nfact:
        arr = arr + ([fact[0]] * fact[1])
    return arr

def left_ideals_of_norm_enumerator(O, N, primitive=True, N_factors=None):
    """
    Given an order `O` and positive integer `N`, returns list of all left ideals of norm `ell`.
    From https://github.com/sagemath/sage/pull/37103, edited to allow for non-prime norm.
    """

    def check_ideal_in_list(arr, I):
        for I2 in arr:
            if I.conjugate().is_equivalent(I2.conjugate()):
                return True
        return False 

    if O.base_ring() != ZZ:
        raise NotImplementedError("only implemented for quaternion algebras over QQ")
    if N not in ZZ:
        raise Exception("N should be integer")
    if N <= 0:
        raise Exception("N should be positive")
    if N == 1:
        yield O.unit_ideal()
        return
    if not N.is_prime():
        Nfact = N_factors
        if Nfact == None:
            Nfact = flattened_factors(N)
        ell = Nfact[0]
        for I1 in left_ideals_of_norm_enumerator(O, ell):
            for I2 in left_ideals_of_norm_enumerator(I1.right_order(), N / ell, N_factors=Nfact[1:]):
                I = I1 * I2
                if primitive:
                    if not is_cyclic(I):
                        continue
                yield I
        return

    # Proceed for prime norm
    ell = N

    if ell == 2:
        # 2 is a special case
        expected_no = 3
        elt = find_element_defining_embedding(O, 1, 0)
        if elt != None:
            expected_no = 2
        already_found = []
        b = O.unit_ideal().reduced_basis()
        b2 = 1000
        while len(already_found) < expected_no:
            elt = randrange(-b2, b2)*b[0] + randrange(-b2, b2)*b[1] + randrange(-b2, b2)*b[2] + randrange(-b2, b2)*b[3]
            if elt == 0:
                continue
            I = O*elt + O*ell
            I = I.intersection(I.left_order()).intersection(I.right_order())
            if I.norm() == ell:
                if not check_ideal_in_list(already_found, I):
                    already_found.append(I)
                    yield I
        return
        
    B = O.quaternion_algebra()
    i, _, _ = B.gens()
    q = -ZZ(i**2)
    if (2 * q) % ell == 0:
        raise NotImplementedError(f"only implemented for odd prime norm not dividing 2i^2(={2*q})")

    mapping = _P1_to_ideals(O, ell)

    yield mapping(0, 1)

    for b in range(ell):
        yield mapping(1, b)

def left_isomorphism(I, J):
    """
    Given two isomorphic left ideals I, J computes α such that J = I*α
    Source: learning to sqi
    """
    B = I.quaternion_algebra()
    if B != J.quaternion_algebra():
        raise ValueError("Arguments must be ideals in the same algebra.")
    if I.left_order() != J.left_order():
        raise ValueError("Arguments must have the same left order.")
    IJ = I.conjugate() * J
    L = minkowski_reduced_basis(IJ)
    for t in L:
        α = t / I.norm()
        if J == I * α:
            return α
    raise ValueError("Could not find a left isomorphism...")

def check_orders_conjugate(O1, O2, certificate=False):
    """
    Check two quaternion orders are equivalent.
    """
    def ideal_test(I1, O0, O2):
        I2 = connecting_ideal(O2, O0)
        is_conj = I1.is_equivalent(I2)
        if not is_conj:
            return False, None
        if not certificate:
            return True, None
        return True, left_isomorphism(I1.conjugate(), I2.conjugate())
    
    B = O1.quaternion_algebra()
    i, j, k = B.gens()
    O0 = B.maximal_order()
    I1 = connecting_ideal(O1, O0)
    found, a = ideal_test(I1, O0, O2)
    if found == False:
        found, a = ideal_test(I1, O0, B.quaternion_order((j.inverse() * O2 * j).basis()))
        if found and a != None: a = j * a
    if found == False:
        found, a = ideal_test(I1, O0, B.quaternion_order((i.inverse() * O2 * i).basis()))
        if found and a != None: a = i * a
    if found == False:
        found, a = ideal_test(I1, O0, B.quaternion_order((k.inverse() * O2 * k).basis()))
        if found and a != None: a = k * a
    if not certificate:
        return found
    return found, a

def all_cornacchia(d, m):
    """
    Returns all solutions to x^2 + dy^2 = m, including imprimitive solutions.
    """
    if m < 0: return []
    if m == 0: return [(0, 0)]
    sols = []
    # Iterate over g such that g^2 divides m
    #   Writing m = q1^e1 * q2^e2 *..., we do this by storing an array [f1, f2, ...] where g = q1^f1 * q2^f2 * ..., and 0 <= fi <= floor(ei / 2)
    #   And we increase it in each iteration [0, 0, 0, ...] -> [1, 0, 0, ...] -> [2, 0, 0, ...] -> (then when f1 is maximum) [0, 1, 0, ...] -> [1, 1, 0, ...] -> ...
    fs = factor(m)
    g_fac_arr = [-1] + [0]*len(fs) # store the exponents of g in this array
    while True:
        g_fac_arr[0] += 1
        g = 1
        for i in range(0, len(fs)):
            if g_fac_arr[i] == floor(fs[i][1]/QQ(2)) + 1: 
                g_fac_arr[i] = 0
                g_fac_arr[i+1] += 1
            # expand value of g
            g *= fs[i][0]**g_fac_arr[i]
        if g_fac_arr[-1] != 0: break
        tempm = ZZ(m/(g**2))
        # Find primitive solutions to x^2 + dy^2 = m/g^2 using Cornacchias and scale them to solution (gx, gy)
        P = PolynomialRing(Integers(tempm), 'X')
        X = P.gens()[0]
        rs =[ZZ(r) for r in (X**2 + d).roots(multiplicities=False)]
        # The above finds solutions r to r^2 = -d  modulo tempm
        #    check if we can add r = tempm, only possible when tempm^2 + d.0^2 = tempm, i.e. tempm^2 = tempm, i.e. tempm = 1
        if tempm == 1:
            rs.append(ZZ(1))
        bound = round(sqrt(tempm),5)
        for r in rs:
            n = tempm
            while r > bound:
                n, r = r, n%r
            s = sqrt((tempm - r**2)/d)
            if s in ZZ:
                sols.extend([(g*r, g*s), (g*r, -g*s), (-g*r, g*s), (-g*r, -g*s)])
                if d == 1:
                    sols.extend([(g*s, g*r), (g*s, -g*r), (-g*s, g*r), (-g*s, -g*r)])
    return list(set(sols))

def find_element_defining_embedding(O, d, t, all_slns=False, filter_func=None):
    """
    Finds an element in quaternion order 'O' with trace 't' and norm 'd'. Set 'all_slns=True' to get all solutions.
        A function can be provided as 'filter_func' which is called when a solution is found to see if it should be counted or not. We use this for filtering primitive solutions.
    """
    slns = [] if all_slns else None
    # Compute the connecting ideal, and find smaller equivalent ideal, to give right order with lower N
    B = O.quaternion_algebra()
    I = B.maximal_order() * O
    J, y = small_equivalent_ideal(I, certificate=True)
    O_new = J.right_order()
    # Put basis in HNF
    basis_hnf = lower_hnf_basis(B, O_new.basis())
    e00, e01, e02, e03 = basis_hnf[0]
    _,   e11, e12, e13 = basis_hnf[1]
    _,   _,   e22, e23 = basis_hnf[2]
    _,   _,   _,   e33 = basis_hnf[3]
    if (e00 == 0) or (e11 == 0) or (e22 == 0) or (e33 == 0):
        return slns
    # Find alpha_0
    alpha_0 = t / (2 * e00)
    if (alpha_0 not in ZZ) or (d not in ZZ):
        return slns
    # Compute a, b, N
    q, p = [ZZ(abs(l)) for l in B.invariants()]
    N = lcm([e.denominator() for e in [e00,e01,e02,e03,e11,e12,e13,e22,e23,e33]])
    N2 = N**2
    # Find residues of alpha_1 mod p
    Fp = GF(p)
    sq_mod_p = Fp(d - (alpha_0 * e00)**2) / Fp(q)
    rt1 = sqrt(sq_mod_p)
    if rt1 not in Fp:
        return slns
    rt2 = -rt1
    residues = [Fp_to_int((rt1 - Fp(alpha_0 * e01)) / Fp(e11)), Fp_to_int((rt2 - Fp(alpha_0 * e01)) / Fp(e11))]
    # compute maximum value of k - for each residue
    temp1 = d - (alpha_0**2)*(e00**2)
    temp1_scaled = N2 * temp1
    temp2 = sqrt(temp1 / q) - alpha_0*e01
    ks = [floor((temp2 - ZZ(r)*e11)/(p*e11)) for r in residues]
    # loop over k decreasing, for each residue
    max_iter = sum([k + 1 for k in ks if k >= 0])
    while max(ks) >= 0:
        k_index = argmax(ks)
        k = ks[k_index]
        r = residues[k_index]
        ks[k_index] = ks[k_index] - 1
        # Compute u and v (v = RHS for Cornacchia)
        alpha_1 = ZZ(r) + k*p
        gamma_1 = alpha_0*e01 + alpha_1*e11
        u = q * N2 * gamma_1**2
        v = ZZ((temp1_scaled - u) / p)
        # find all solutions to Cornacchia's
        betas = all_cornacchia(q, v)
        for beta_pair in betas:
            # Check if this gives a solution with integral alpha_2 and alpha_3
            alpha_2 = (beta_pair[0] - N*alpha_1*e12 - N*alpha_0*e02) / (N*e22)
            alpha_3 = (beta_pair[1] - N*alpha_1*e13 - N*alpha_2*e23 - N*alpha_0*e03) / (N*e33)
            if (alpha_2 in ZZ) and (alpha_3 in ZZ):
                alpha = alpha_0*basis_hnf[0] + alpha_1*basis_hnf[1] + alpha_2*basis_hnf[2] + alpha_3*basis_hnf[3]
                alpha_in_O = y * alpha * y**(-1) # map alpha back in to original order
                valid_sln = True
                if filter_func != None:
                    valid_sln = filter_func(alpha_in_O, k)
                if valid_sln:
                    if all_slns: slns.append(alpha_in_O)
                    if not all_slns: return alpha_in_O
    return slns

def eval_isomorphism(alpha, B, gamma):
    """
    Evaluates a quaternion in an isomorphism of quaternion algebras
    """
    i, j, k = B.gens()
    return sum([coeff*b for coeff, b in zip(alpha.coefficient_tuple(), [1, i*gamma, j, k*gamma])])

def factors_easily(n, B=2**20):
    """
    Given a number n, checks if a n is "Cornacchia Friendly" (= easily factorable)
    """
    n = ZZ(n)
    if n < 0: return False
    if n < 2**160: return True
    l,_ = n.factor(limit=B)[-1]
    return l < 2**160 or is_pseudoprime(l)

def isomorphism_gamma(B_old, B):
    """
    Defines an isomorphism of quaternion algebras, See Lemma 10 [EPSV23]
    """
    if B_old == B:
        return B(1), B(1)
    i_old, j_old, k_old = B_old.gens()
    q_old = -ZZ(i_old**2)
    i, j, k = B.gens()
    q = -ZZ(i**2) 
    p = -ZZ(j**2)
    x, y = DiagonalQuadraticForm(QQ, [1,p]).solve(q_old/q)
    return x + j*y, (x + j_old*y)**(-1)

def quat_algs(p):
    """
    Generate 3 isomorphic quaternion algebras ramified at p \neq 2 and infity, with abs(i^2) small
    """
    Bs = []
    mod = 4
    if p % 4 == 3:
        Bs.append(QuaternionAlgebra(-1, -p))
    q = 1
    while len(Bs) < 3:
        q = next_prime(q)
        if (-q) % mod == 1 and kronecker(-q, p) == -1:
            Bs.append(QuaternionAlgebra(-q, -p))
    assert all([B.ramified_primes() == [p] for B in Bs ])
    return Bs

def find_element_defining_embedding_randomized(O, d, t, filter_func=None, debug=False):
    """
        Continuously randomizes basis for the order O, until an element of trace t norm d can be found without doing any hard factorizations.
            Maps the element back into the starting order and returns it.
        
        Returns two values:
        - The element of trace t norm d.
        - A "confidence" boolean.
            If no element found, True if we're certain it's not possible an element exists. False if it still might be possible as we may have skipped it.
    """
    def find_element_defining_embedding_with_skips(O, d, t):
        """
        Attempts to find an element in quaternion order 'O' with trace 't' and norm 'd', but may miss solutions.
            In solving x^2+|a|y^2=v with Coracchias, skips if v if it is hard to factor.
            Returns solution, and 'confidence' boolean that is True if no solutions have been skipped.
        """

        # Put basis in HNF
        basis_hnf = lower_hnf_basis(B, O.basis())
        e00, e01, e02, e03 = basis_hnf[0]
        _,   e11, e12, e13 = basis_hnf[1]
        _,   _,   e22, e23 = basis_hnf[2]
        _,   _,   _,   e33 = basis_hnf[3]
        if (e00 == 0) or (e11 == 0) or (e22 == 0) or (e33 == 0):
            return None, True
        # Find alpha_0
        alpha_0 = t / (2 * e00)
        if (alpha_0 not in ZZ) or (d not in ZZ):
            # If none works, we're confident no solution exists in any conjugate order
            return None, True
        # Compute N
        N = lcm([e.denominator() for e in [e00,e01,e02,e03,e11,e12,e13,e22,e23,e33]])
        N2 = N**2
        # Find residues of alpha_1 mod p
        Fp = GF(p)
        sq_mod_p = Fp(d - (alpha_0 * e00)**2) / Fp(q)
        rt1 = sqrt(sq_mod_p)
        if rt1 not in Fp:
            return None, True
        rt2 = -rt1
        residues = [Fp_to_int((rt1 - Fp(alpha_0 * e01)) / Fp(e11)), Fp_to_int((rt2 - Fp(alpha_0 * e01)) / Fp(e11))]
        # compute maximum value of k - for each residue
        temp1 = d - (alpha_0**2)*(e00**2)
        temp1_scaled = N2 * temp1
        temp2 = sqrt(temp1 / q) - alpha_0*e01
        ks = [floor((temp2 - ZZ(r)*e11)/(p*e11)) for r in residues]
        # loop over k decreasing, for each residue
        max_iter = sum([k + 1 for k in ks if k >= 0])
        skipped_v = False
        while max(ks) >= 0:
            k_index = argmax(ks)
            k = ks[k_index]
            r = residues[k_index]
            ks[k_index] = ks[k_index] - 1
            # Compute u and v (v = RHS for Cornacchia)
            alpha_1 = ZZ(r) + k*p
            gamma_1 = alpha_0*e01 + alpha_1*e11
            u = q * N2 * gamma_1**2
            v = ZZ((temp1_scaled - u) / p)
            if factors_easily(v):
                # find all solutions to Cornacchia's
                betas = all_cornacchia(q, v)
                for beta_pair in betas:
                    # Check if this gives a solution with integral alpha_2 and alpha_3
                    alpha_2 = (beta_pair[0] - N*alpha_1*e12 - N*alpha_0*e02) / (N*e22)
                    alpha_3 = (beta_pair[1] - N*alpha_1*e13 - N*alpha_2*e23 - N*alpha_0*e03) / (N*e33)
                    if (alpha_2 in ZZ) and (alpha_3 in ZZ):
                        alpha = alpha_0*basis_hnf[0] + alpha_1*basis_hnf[1] + alpha_2*basis_hnf[2] + alpha_3*basis_hnf[3]
                        valid_sln = True
                        if filter_func != None:
                            valid_sln = filter_func(alpha, k)
                        if valid_sln:
                            return alpha, (not skipped_v)
            else:
                skipped_v = True
        # If we didn't skip any v's we know no solution exists
        return None, (not skipped_v)

    p = -ZZ(O.quaternion_algebra().gens()[1]**2)
    Bs = quat_algs(p)
    for B in Bs:
        # The maximal order with small denominator O_0
        O0 = B.maximal_order()
        # Compute isomorphism between the quat algebras
        gamma, gamma_inv = isomorphism_gamma(O.quaternion_algebra(), B)
        # Transfer the maximal order to new quaternion algebra
        O_in_new_quatalg = B.quaternion_order([eval_isomorphism(alpha, B, gamma) for alpha in O.gens()])
        if debug:
            print(f"\nFinding solution in {O_in_new_quatalg}")
        q, p = [ZZ(abs(l)) for l in B.invariants()]
        # Find connecting ideal
        I = O0 * O_in_new_quatalg
        I = I * denominator(I.norm())
        # Reduced basis to find other small equivalent ideals, which gives suitable isomorphisms of O
        basis_hnf = lower_hnf_basis(B, I.basis())
        M = matrix(QQ, [ai.coefficient_tuple() for ai in basis_hnf])
        S = 2**ceil(log(p*q, 2))
        D = diagonal_matrix(round(S * sqrt(g.reduced_norm())) for g in B.basis())
        reduced_basis = (M * D).LLL() * ~D
        # Define constants for conjugating order
        used = []
        max_size = round(p**(1/1.8),5) + 10
        bound = max(round(log(p,2)/10), 10)
        # Try a bunch of small connecting ideals
        for (a1,a2,a3,a4) in itertools.product(range(0,bound+1), range(-bound,bound+1), range(-bound,bound+1), range(-bound,bound+1)):
            if a1 == a2 == a3 == a4 == 0:
                continue
            coeffvec = vector(QQ, [a1,a2,a3,a4])
            y = coeffvec * reduced_basis * vector(B.basis())
            Jnorm = y.reduced_norm() / I.norm()
            if y in used or Jnorm > max_size:
                continue
            used.append(y)
            y = y.conjugate() / I.norm()
            J = I * y
            JRO = J.right_order()
            beta, confidence = find_element_defining_embedding_with_skips(JRO, d, t)
            if beta:
                beta_new =  y * beta * y**(-1)
                return eval_isomorphism(beta_new, O.quaternion_algebra(), gamma_inv)
            else:
                if confidence: return None, True
    return None, False

class QuaternionLattice:
    # add equality check, scalar multiplication, multiplication by quaternion, negation -L
    def __init__(self, thing, quatalg=None):
        thing2 = thing
        if quatalg != None:
            if str(type(quatalg)) != "<class 'sage.algebras.quatalg.quaternion_algebra.QuaternionAlgebra_ab_with_category'>":
                raise Exception("Parameter `quatalg` is not a quaternion algebra.")
        if type(thing) is sage.matrix.matrix_integer_dense.Matrix_integer_dense:
            thing2 = matrix(QQ, thing)
        if type(thing2) is sage.matrix.matrix_rational_dense.Matrix_rational_dense:
            if quatalg == None:
                raise Exception("Needs to know quaternion algebra with `quatalg` arguement.")
            if len(thing2.rows()) != 4:
                raise Exception("Basis matrix should have 4 rows")
            reduced = lower_hnf_matrix(thing2)
            reduced_cleaned = column_matrix([c for c in reduced.columns() if c.list() != [0, 0, 0, 0]])
            self.quatalg = quatalg
            self.basis_matrix = reduced_cleaned
        else:
            if str(type(thing)) == "<class 'sage.algebras.quatalg.quaternion_algebra.QuaternionOrder_with_category'>" or str(type(thing)) == "<class 'sage.algebras.quatalg.quaternion_algebra.QuaternionFractionalIdeal_rational'>":
                reduced = lower_hnf_matrix(basis_to_matrix(thing.basis()))
                reduced_cleaned = column_matrix([c for c in reduced.columns() if c.list() != [0, 0, 0, 0]])
                self.basis_matrix = reduced_cleaned
                self.quatalg = thing.quaternion_algebra()
            else:
                if type(thing) is list:
                    if thing == []: raise Exception("basis is empty")
                    quatalg = parent(thing[-1])
                    if str(type(quatalg)) != "<class 'sage.algebras.quatalg.quaternion_algebra.QuaternionAlgebra_ab_with_category'>":
                        raise Exception("Basis entries are not in a quaternion algebra.")
                    self.quatalg = quatalg
                    reduced = lower_hnf_matrix(basis_to_matrix(thing))
                    reduced_cleaned = column_matrix([c for c in reduced.columns() if c.list() != [0, 0, 0, 0]])
                    self.basis_matrix = reduced_cleaned
                else:
                    raise Exception("unknown type")
    def __add__(self, o):
        if self.quatalg != o.quatalg:
            raise Exception("Adding quaternion lattices in different algebras")
        joint_basis_matrix = column_matrix(self.basis_matrix.columns() + o.basis_matrix.columns())
        return QuaternionLattice(joint_basis_matrix, quatalg=self.quatalg)
    def __mul__(self, o):
        if self.quatalg != o.quatalg:
            raise Exception("Multiplying quaternion lattices in different algebras")
        newbasis = [e1*e2 for e1 in self.basis() for e2 in o.basis()]
        return QuaternionLattice(basis_to_matrix(newbasis), quatalg=self.quatalg)
    def intersect(self, o):
        if self.quatalg != o.quatalg:
            raise Exception("Intersecting quaternion lattices in different algebras")
        V = self.quatalg.base_ring()**4
        B = V.span([V(list(g)) for g in self.basis()], ZZ)
        C = V.span([V(list(g)) for g in o.basis()], ZZ)
        newmat = column_matrix([list(e) for e in B.intersection(C).basis()])
        return QuaternionLattice(newmat, quatalg=self.quatalg)
    def intersection(self, o):
        return self.intersect(o)
    def basis(self):
        return matrix_to_basis(self.quatalg, self.basis_matrix)
    def lower_hnf_basis(self):
        return self.basis()
    def upper_hnf_basis(self):
        return matrix_to_basis(self.quatalg, upper_hnf_matrix(self.basis_matrix))
    def to_ideal(self):
        return self.quatalg.ideal(self.basis())
    def to_order(self):
        return self.quatalg.quaternion_order(self.basis())

def basis_to_matrix(basis, R=QQ):
    """
    Given a basis of an ideal as a list, convert it to a column-style basis matrix.
    Assumes all basis elements are of correct type - either QuaternionAlgebra or NumberField over the rationals.
    If over a PolynomialRing instead of rationals, pass in the ring as `R`.
    """
    B = parent(basis[len(basis)-1])
    if str(type(parent(basis[0]))) == "<class 'sage.rings.number_field.number_field.NumberField_quadratic_with_category'>":
        basis_len = 2
    else:
        basis_len = len(parent(basis[len(basis)-1]).basis())
    return matrix(R,[[B(basis[m])[r] for r in range(0, basis_len)] for m in range(0, len(basis))]).transpose()

def matrix_to_basis(B, M):
    """
    Given a quaternion algebra B, and a column-style basis matrix M, return the basis as elements of B.
    """
    return [a[0] for a in (M.transpose() * matrix([B(1)] + list(B.gens())).transpose())]

def lower_hnf_matrix(M, transformation=False):
    """
    Reduces matrix to lower triangular form.
    If tranformation=True, also returns matrix T such that M.T is lower HNF.
    """
    denom = lcm([l.denominator() for l in M.list()])
    # M_ZZ = matrix(ZZ,[[M[k][l]*denom for l in range(0, M.dimensions()[1])] for k in range(0, M.dimensions()[0])])
    M_ZZ = matrix(ZZ, [[M[k][l]*denom for l in range(0, M.dimensions()[1])] for k in range(0, M.dimensions()[0])])
    if not transformation:
        return (1/denom) * (M_ZZ.transpose().hermite_form().transpose())
    res1, trans = M_ZZ.transpose().hermite_form(transformation=True)
    return (1/denom) * (res1.transpose()), trans.transpose()

def upper_hnf_matrix(M, transformation=False):
    """
    Reduces matrix to upper triangular form.
    If tranformation=True, also returns matrix T such that M.T is upper HNF.
    """
    if not transformation:
        return lower_hnf_matrix(M[::-1,:])[::-1,::-1]
    res, trans = lower_hnf_matrix(M[::-1,:], True)
    return res[::-1,::-1], trans * identity_matrix(trans.dimensions()[1])[::-1,:]

def lower_hnf_basis(B, basis, R=QQ):
    """
    Reduces basis to lower triangular form.
    If over a PolynomialRing instead of rationals, pass in the ring as `R`.
    """
    return matrix_to_basis(B, lower_hnf_matrix(basis_to_matrix(basis, R)))

def upper_hnf_basis(B, basis, R=QQ):
    """
    Reduces basis to upper triangular form.
    If over a PolynomialRing instead of rationals, pass in the ring as `R`.
    """
    return matrix_to_basis(B, upper_hnf_matrix(basis_to_matrix(basis, R)))