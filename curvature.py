import sympy as sp
from collections import defaultdict
from itertools import product
from argparse import ArgumentParser


def nodes():
    '''
    Returns a dictionary of variables representing the nodes entering the calculation
    '''

    nodes = sp.symbols(
                'v v_x^+ v_x^- v_y^+ v_y^- v_z^+ v_z^- '
                'v_xy^++ v_xy^+- v_xy^-+ v_xy^-- '
                'v_xz^++ v_xz^+- v_xz^-+ v_xz^-- '
                'v_yz^++ v_yz^+- v_yz^-+ v_yz^-- '
            )

    node_labels = (
            '0', 'x+', 'x-', 'y+', 'y-', 'z+', 'z-',
            'xy++', 'xy+-', 'xy-+', 'xy--',
            'xz++', 'xz+-', 'xz-+', 'xz--',
            'yz++', 'yz+-', 'yz-+', 'yz--',
        )

    return {label: node for node, label in zip(nodes, node_labels)}


def edges():
    '''
    Returns a dictionary of variables representing the edges entering the calculation
    '''

    edges = sp.symbols(
                'e_x^+ e_x^- e_y^+ e_y^- e_z^+ e_z^- '
                'e_xy^++ e_xy^+- e_xy^-+ e_xy^-- '
                'e_yx^++ e_yx^+- e_yx^-+ e_yx^-- '
                'e_xz^++ e_xz^+- e_xz^-+ e_xz^-- '
                'e_zx^++ e_zx^+- e_zx^-+ e_zx^-- '
                'e_yz^++ e_yz^+- e_yz^-+ e_yz^-- '
                'e_zy^++ e_zy^+- e_zy^-+ e_zy^-- '
            )

    edge_labels = (
            'x+', 'x-', 'y+', 'y-', 'z+', 'z-',
            'xy++', 'xy+-', 'xy-+', 'xy--',
            'yx++', 'yx+-', 'yx-+', 'yx--',
            'xz++', 'xz+-', 'xz-+', 'xz--',
            'zx++', 'zx+-', 'zx-+', 'zx--',
            'yz++', 'yz+-', 'yz-+', 'yz--',
            'zy++', 'zy+-', 'zy-+', 'zy--',
        )

    return {label: edge for edge, label in zip(edges, edge_labels)}


def fill_permutations(array, symmetric=False):
    '''
    Fills in the remaining entries of an array whose entries are related by cyclic
    permutations of x, y and z, after a complete set of independent entries, from which
    the remaining entries can be deduced, has been filled.

    If 'symmetric=True' is passed, missing entries are also filled using the assumption
    that the array is symmetric in the first two indices.
    '''

    def substitute(expr, subs):
        '''
        Performs a group of simultaneous substitutions consistently, by first
        substituting each original variable with a dummy variable, and then substituting
        the corresponding target value for each dummy variable.

        The 'subs' method of SymPy expressions either is buggy, or I don't understand
        the logic by which it operates, as shown by the following example:

            x, y, z = sp.symbols('x y z')
            f, g, h = sp.symbols('f g h', cls=sp.Function, commutative=True)

            expr = f(x) + g(y) + h(z)
            subs_x = [(x, y), (y, z), (z, x)]
            subs_f = [(f, g), (g, h), (h, f)]

            expr.subs(subs_x, simultaneous=True)    # gives f(y) + g(z) + h(x)
            expr.subs(subs_f, simultaneous=True)    # gives f(x) + g(y) + h(z) !!
        '''
        variables = [s[0] for s in subs]
        values = [s[1] for s in subs]
        dummies = [sp.symbols(f'dummy{i}', cls=type(x), commutative=x.is_commutative)
                   for i, x in enumerate(variables)]

        first = [(x, d) for x, d in zip(variables, dummies)]
        second = [(d, v) for d, v in zip(dummies, values)]

        return expr.subs(first).subs(second)

    # (x, y, z) -> (y, z, x)
    subs_1 = [
        (p_x, p_y), (p_y, p_z), (p_z, p_x),
        (v['x+'], v['y+']), (v['x-'], v['y-']),
        (v['y+'], v['z+']), (v['y-'], v['z-']),
        (v['z+'], v['x+']), (v['z-'], v['x-']),
        (v['xy++'], v['yz++']), (v['xy+-'], v['yz+-']),
        (v['xy-+'], v['yz-+']), (v['xy--'], v['yz--']),
        (v['yz++'], v['xz++']), (v['yz+-'], v['xz+-']),
        (v['yz-+'], v['xz-+']), (v['yz--'], v['xz--']),
        (v['xz++'], v['xy++']), (v['xz+-'], v['xy+-']),
        (v['xz-+'], v['xy-+']), (v['xz--'], v['xy--']),
        (e['x+'], e['y+']), (e['x-'], e['y-']),
        (e['y+'], e['z+']), (e['y-'], e['z-']),
        (e['z+'], e['x+']), (e['z-'], e['x-']),
        (e['xy++'], e['yz++']), (e['xy+-'], e['yz+-']),
        (e['xy-+'], e['yz-+']), (e['xy--'], e['yz--']),
        (e['yz++'], e['zx++']), (e['yz+-'], e['zx+-']),
        (e['yz-+'], e['zx-+']), (e['yz--'], e['zx--']),
        (e['zx++'], e['xy++']), (e['zx+-'], e['xy+-']),
        (e['zx-+'], e['xy-+']), (e['zx--'], e['xy--']),
        (e['yx++'], e['zy++']), (e['yx+-'], e['zy+-']),
        (e['yx-+'], e['zy-+']), (e['yx--'], e['zy--']),
        (e['zy++'], e['xz++']), (e['zy+-'], e['xz+-']),
        (e['zy-+'], e['xz-+']), (e['zy--'], e['xz--']),
        (e['xz++'], e['yx++']), (e['xz+-'], e['yx+-']),
        (e['xz-+'], e['yx-+']), (e['xz--'], e['yx--']),
    ]

    # (x, y, z) -> (z, x, y)
    subs_2 = [
        (p_x, p_z), (p_y, p_x), (p_z, p_y),
        (v['x+'], v['z+']), (v['x-'], v['z-']),
        (v['y+'], v['x+']), (v['y-'], v['x-']),
        (v['z+'], v['y+']), (v['z-'], v['y-']),
        (v['xy++'], v['xz++']), (v['xy+-'], v['xz+-']),
        (v['xy-+'], v['xz-+']), (v['xy--'], v['xz--']),
        (v['yz++'], v['xy++']), (v['yz+-'], v['xy+-']),
        (v['yz-+'], v['xy-+']), (v['yz--'], v['xy--']),
        (v['xz++'], v['yz++']), (v['xz+-'], v['yz+-']),
        (v['xz-+'], v['yz-+']), (v['xz--'], v['yz--']),
        (e['x+'], e['z+']), (e['x-'], e['z-']),
        (e['y+'], e['x+']), (e['y-'], e['x-']),
        (e['z+'], e['y+']), (e['z-'], e['y-']),
        (e['xy++'], e['zx++']), (e['xy+-'], e['zx+-']),
        (e['xy-+'], e['zx-+']), (e['xy--'], e['zx--']),
        (e['yz++'], e['xy++']), (e['yz+-'], e['xy+-']),
        (e['yz-+'], e['xy-+']), (e['yz--'], e['xy--']),
        (e['zx++'], e['yz++']), (e['zx+-'], e['yz+-']),
        (e['zx-+'], e['yz-+']), (e['zx--'], e['yz--']),
        (e['yx++'], e['xz++']), (e['yx+-'], e['xz+-']),
        (e['yx-+'], e['xz-+']), (e['yx--'], e['xz--']),
        (e['zy++'], e['yx++']), (e['zy+-'], e['yx+-']),
        (e['zy-+'], e['yx-+']), (e['zy--'], e['yx--']),
        (e['xz++'], e['zy++']), (e['xz+-'], e['zy+-']),
        (e['xz-+'], e['zy-+']), (e['xz--'], e['zy--']),
    ]

    # All (multi-)indices of the array
    idx = list(product(*[range(N) for N in array.shape]))
    # Indices containing a non-zero entry
    idx_nonzero = [i for i in idx if array[i] != 0]

    # If 'symmetric=True', fill in the values array[b, a, ...] based on the already
    # entered values array[a, b, ...] and update the list of non-zero indices
    if symmetric:
        idx_sym = [(i[1], i[0]) + i[2:] for i in idx_nonzero]
        for i, k in zip(idx_nonzero, idx_sym):
            array[k] = array[i]
        idx_nonzero.extend(i for i in idx_sym if i not in idx_nonzero)

    # (0, 1, 2) -> (1, 2, 0)
    idx_1 = [tuple(map(lambda x: (x + 1) % 3, i)) for i in idx_nonzero]
    # (0, 1, 2) -> (2, 0, 1)
    idx_2 = [tuple(map(lambda x: (x + 2) % 3, i)) for i in idx_nonzero]

    extension = sp.MutableDenseNDimArray(sp.zeros(len(array)), array.shape)

    for i_new, i in zip(idx_1, idx_nonzero):
        extension[i_new] = substitute(array[i], subs_1)

    for i_new, i in zip(idx_2, idx_nonzero):
        extension[i_new] = substitute(array[i], subs_2)

    return array + extension


def derivatives():
    r'''
    Returns the arrays

        DE[a, i, b]
        DE_[a, i, b]
        D2E[a, b, i, c]
        DV[a]
        D2V[a, b]

    representing respectively the reduced operators corresponding to

        \Delta_a E_i(S^b, v)        Eq. (4.5)
        \Delta_a {\cal E}_b^i(v)    Eq. (4.50)
        \Delta_{ab} E_i(S^c, v)     Eqs. (4.37) and (4.47)
        \Delta_a V(v)^2             Eq. (3.22)
        \Delta_{ab} V(v)^2          Eqs. (3.23) and (3.24)

    (equation numbers refer to 2211.04826)
    '''

    def S(e):
        # S(e) is shorthand for the sum d_+(e) + d_-(e)
        # c(e) = 1/2 * S(e) corresponds to c^{(1)}(e) in the notation of 2412.01375
        return 2 * c(e)

    def D(e):
        # D(e) is shorthand for the difference d_+(e) - d_-(e)
        # s(e) = 1/(2i) * D(e) corresponds to s^{(1)}(e) in the notation of 2412.01375
        return 2*I * s(e)

    # DE[a, i, b] = \Delta_a E_i(S^b, v)
    DE = sp.MutableDenseNDimArray(sp.zeros(27), (3, 3, 3))

    # (4.27)
    DE[2, 0, 0] = _1/4 * S(e['z+']) * p_x(v['z+']) - _1/4 * S(e['z-']) * p_x(v['z-'])
    # (4.28)
    DE[2, 1, 0] = -I/4 * D(e['z+']) * p_x(v['z+']) - I/4 * D(e['z-']) * p_x(v['z-'])
    # (4.30)
    DE[2, 0, 1] = I/4 * D(e['z+']) * p_y(v['z+']) + I/4 * D(e['z-']) * p_y(v['z-'])
    # (4.31)
    DE[2, 1, 1] = _1/4 * S(e['z+']) * p_y(v['z+']) - _1/4 * S(e['z-']) * p_y(v['z-'])
    # (4.35)
    DE[2, 2, 2] = _1/2 * (p_z(v['z+']) - p_z(v['z-']))

    DE = fill_permutations(DE)

    # DE_[a, i, b] = \Delta_a {\cal E}_b^i(v)
    DE_ = sp.MutableDenseNDimArray(sp.zeros(27), (3, 3, 3))

    # (4.57)
    DE_[2, 0, 0] = _1/4 * S(e['z+']) / p_x(v['z+']) - _1/4 * S(e['z-']) / p_x(v['z-'])
    # (4.58) [typo: overall sign]
    DE_[2, 1, 0] = -I/4 * D(e['z+']) / p_x(v['z+']) - I/4 * D(e['z-']) / p_x(v['z-'])
    # (4.60) [typo: overall sign]
    DE_[2, 0, 1] = I/4 * D(e['z+']) / p_y(v['z+']) + I/4 * D(e['z-']) / p_y(v['z-'])
    # (4.61)
    DE_[2, 1, 1] = _1/4 * S(e['z+']) / p_y(v['z+']) - _1/4 * S(e['z-']) / p_y(v['z-'])
    # (4.65)
    DE_[2, 2, 2] = _1/2 * (1 / p_z(v['z+']) - 1 / p_z(v['z-']))

    DE_ = fill_permutations(DE_)

    # D2E[a, b, i, c] = \Delta_{ab}E_i(S^c, v)
    D2E = sp.MutableDenseNDimArray(sp.zeros(81), (3, 3, 3, 3))

    # (4.38)
    D2E[2, 2, 0, 0] = _1/2 * S(e['z+']) * p_x(v['z+']) - 2 * p_x(v['0']) \
                      + _1/2 * S(e['z-']) * p_x(v['z-'])
    # (4.39)
    D2E[2, 2, 1, 0] = -I/2 * D(e['z+']) * p_x(v['z+']) \
                      + I/2 * D(e['z-']) * p_x(v['z-'])
    # (4.41)
    D2E[2, 2, 0, 1] = I/2 * D(e['z+']) * p_y(v['z+']) \
                      - I/2 * D(e['z-']) * p_y(v['z-'])
    # (4.42)
    D2E[2, 2, 1, 1] = _1/2 * S(e['z+']) * p_y(v['z+']) - 2 * p_y(v['0']) \
                      + _1/2 * S(e['z-']) * p_y(v['z-'])
    # (4.46)
    D2E[2, 2, 2, 2] = p_z(v['z+']) - 2 * p_z(v['0']) + p_z(v['z-'])
    # (B.19)
    D2E[0, 1, 0, 0] = _1/16 * (S(e['yx++']) + S(e['y+'])) * p_x(v['xy++']) \
                      - _1/16 * (S(e['yx+-']) + S(e['y-'])) * p_x(v['xy+-']) \
                      - _1/16 * (S(e['yx-+']) + S(e['y+'])) * p_x(v['xy-+']) \
                      + _1/16 * (S(e['yx--']) + S(e['y-'])) * p_x(v['xy--'])
    # (B.20)
    D2E[0, 1, 1, 0] = -_1/32 * D(e['yx++']) * D(e['x+']) * p_x(v['xy++']) \
                      - _1/32 * D(e['yx+-']) * D(e['x+']) * p_x(v['xy+-']) \
                      - _1/32 * D(e['yx-+']) * D(e['x-']) * p_x(v['xy-+']) \
                      - _1/32 * D(e['yx--']) * D(e['x-']) * p_x(v['xy--'])
    # (B.21)
    D2E[0, 1, 2, 0] = \
            I/32 * (D(e['yx++']) * S(e['x+']) + 2 * D(e['y+'])) * p_x(v['xy++']) \
            + I/32 * (D(e['yx+-']) * S(e['x+']) + 2 * D(e['y-'])) * p_x(v['xy+-']) \
            - I/32 * (D(e['yx-+']) * S(e['x-']) + 2 * D(e['y+'])) * p_x(v['xy-+']) \
            - I/32 * (D(e['yx--']) * S(e['x-']) + 2 * D(e['y-'])) * p_x(v['xy--'])
    # (B.22)
    D2E[0, 1, 0, 1] = -_1/32 * D(e['xy++']) * D(e['y+']) * p_y(v['xy++']) \
                      - _1/32 * D(e['xy+-']) * D(e['y-']) * p_y(v['xy+-']) \
                      - _1/32 * D(e['xy-+']) * D(e['y+']) * p_y(v['xy-+']) \
                      - _1/32 * D(e['xy--']) * D(e['y-']) * p_y(v['xy--'])
    # (B.23)
    D2E[0, 1, 1, 1] = _1/16 * (S(e['xy++']) + S(e['x+'])) * p_y(v['xy++']) \
                      - _1/16 * (S(e['xy+-']) + S(e['x+'])) * p_y(v['xy+-']) \
                      - _1/16 * (S(e['xy-+']) + S(e['x-'])) * p_y(v['xy-+']) \
                      + _1/16 * (S(e['xy--']) + S(e['x-'])) * p_y(v['xy--'])
    # (B.24)
    D2E[0, 1, 2, 1] = \
            -I/32 * (2 * D(e['x+']) + D(e['xy++']) * S(e['y+'])) * p_y(v['xy++']) \
            + I/32 * (2 * D(e['x+']) + D(e['xy+-']) * S(e['y-'])) * p_y(v['xy+-']) \
            - I/32 * (2 * D(e['x-']) + D(e['xy-+']) * S(e['y+'])) * p_y(v['xy-+']) \
            + I/32 * (2 * D(e['x-']) + D(e['xy--']) * S(e['y-'])) * p_y(v['xy--'])
    # (B.25)
    D2E[0, 1, 0, 2] = \
            -I/32 * (2 * D(e['yx++']) + S(e['xy++']) * D(e['y+'])) * p_z(v['xy++']) \
            - I/32 * (2 * D(e['yx+-']) + S(e['xy+-']) * D(e['y-'])) * p_z(v['xy+-']) \
            + I/32 * (2 * D(e['yx-+']) + S(e['xy-+']) * D(e['y+'])) * p_z(v['xy-+']) \
            + I/32 * (2 * D(e['yx--']) + S(e['xy--']) * D(e['y-'])) * p_z(v['xy--'])
    # (B.26) [typo: p_y -> p_z]
    D2E[0, 1, 1, 2] = \
            I/32 * (D(e['yx++']) * S(e['x+']) + 2 * D(e['xy++'])) * p_z(v['xy++']) \
            - I/32 * (D(e['yx+-']) * S(e['x+']) + 2 * D(e['xy+-'])) * p_z(v['xy+-']) \
            + I/32 * (D(e['yx-+']) * S(e['x-']) + 2 * D(e['xy-+'])) * p_z(v['xy-+']) \
            - I/32 * (D(e['yx--']) * S(e['x-']) + 2 * D(e['xy--'])) * p_z(v['xy--'])
    # (B.27)
    D2E[0, 1, 2, 2] = _1/32 * S(e['yx++']) * S(e['x+']) * p_z(v['xy++']) \
                      + _1/32 * S(e['yx++']) * S(e['y+']) * p_z(v['xy++']) \
                      - _1/32 * S(e['yx+-']) * S(e['x+']) * p_z(v['xy+-']) \
                      - _1/32 * S(e['yx+-']) * S(e['y-']) * p_z(v['xy+-']) \
                      - _1/32 * S(e['yx-+']) * S(e['x-']) * p_z(v['xy-+']) \
                      - _1/32 * S(e['yx-+']) * S(e['y+']) * p_z(v['xy-+']) \
                      + _1/32 * S(e['yx--']) * S(e['x-']) * p_z(v['xy--']) \
                      + _1/32 * S(e['yx--']) * S(e['y-']) * p_z(v['xy--']) \

    D2E = fill_permutations(D2E, symmetric=True)

    # DV[a] = \Delta_a V(v)^2 / V(v)^2
    DV = sp.MutableDenseNDimArray(sp.zeros(3, 1), (3, ))

    DV[0] = _1/2 * (V(v['x+'])**2 - V(v['x-'])**2) / V(v['0'])**2

    DV = fill_permutations(DV)

    # D2V[a, b] = \Delta_{ab} V(v)^2 / V(v)^2
    D2V = sp.MutableDenseNDimArray(sp.zeros(9), (3, 3))

    D2V[0, 0] = (V(v['x+'])**2 - 2*V(v['0'])**2 + V(v['x-'])**2) / V(v['0'])**2
    D2V[0, 1] = _1/4 * (V(v['xy++'])**2 - V(v['xy++'])**2
                        - V(v['xy++'])**2 + V(v['xy++'])**2) / V(v['0'])**2

    D2V = fill_permutations(D2V, symmetric=True)

    return DE, DE_, D2E, DV, D2V


def curvature(term=None):
    '''
    Returns an expression representing the operator obtained from Eq. (3.2) of
    2211.04826 upon substituting the individual operators entering Eq. (3.2) with their
    reduced counterparts
    '''

    # (4.2)
    E = sp.diag(p_x(v['0']), p_y(v['0']), p_z(v['0']))
    # (4.3)
    E_ = sp.diag(1/p_x(v['0']), 1/p_y(v['0']), 1/p_z(v['0']))

    # (3.3)
    Q = sp.tensorcontraction(sp.tensorproduct(E, E), (1, 3))
    # (3.4)
    Q_ = sp.tensorcontraction(sp.tensorproduct(E_, E_), (1, 3))

    DE, DE_, D2E, DV, D2V = derivatives()

    # (3.5)
    A = sp.tensorcontraction(sp.tensorproduct(E, DE), (1, 3))
    A = sp.permutedims(A, (0, 2, 1))
    # (3.6)
    B = sp.tensorcontraction(sp.tensorproduct(E_, DE), (1, 3))

    # The terms on the RHS of Eq. (3.2)
    terms = [
         sp.tensorcontraction(sp.tensorproduct(E, D2E),
                           (0, 2), (1, 4), (3, 5)),

         sp.tensorcontraction(sp.tensorproduct(Q, E_, D2E),
                           (0, 4), (1, 5), (2, 7), (3, 6)),

         sp.tensorcontraction(sp.tensorproduct(DE, DE),
                           (0, 2), (1, 4), (3, 5)),

         sp.tensorcontraction(sp.tensorproduct(DE, DE),
                           (0, 5), (1, 4), (2, 3)),

         sp.tensorcontraction(sp.tensorproduct(Q, DE, DE_),
                           (0, 2), (1, 5), (3, 6), (4, 7)),

         sp.tensorcontraction(sp.tensorproduct(Q, Q_, DE, DE),
                           (0, 4), (1, 7), (2, 6), (3, 9), (5, 8)),

         sp.tensorcontraction(sp.tensorproduct(A, B),
                           (0, 2), (1, 4), (3, 5)),

         sp.tensorcontraction(sp.tensorproduct(A, B),
                           (0, 4), (1, 2), (3, 5)),

         sp.tensorcontraction(sp.tensorproduct(A, B),
                           (0, 4), (1, 3), (2, 5)),

         sp.tensorcontraction(sp.tensorproduct(Q_, A, A),
                           (0, 3), (1, 6), (2, 7), (4, 5)),

         sp.tensorcontraction(sp.tensorproduct(Q, B, B),
                           (0, 3), (1, 6), (2, 4), (5, 7)),

         sp.tensorcontraction(sp.tensorproduct(Q, B, DV),
                           (0, 3), (1, 5), (2, 4)),

         sp.tensorcontraction(sp.tensorproduct(A, DV), (0, 2), (1, 3)),

         sp.tensorcontraction(sp.tensorproduct(A, DV), (0, 3), (1, 2)),

         sp.tensorcontraction(sp.tensorproduct(Q, DV, DV), (0, 2), (1, 3)),

         sp.tensorcontraction(sp.tensorproduct(Q, D2V), (0, 2), (1, 3))
         ]

    # Coefficients of each term
    coefficients = [-2, 2, -1, -sp.S(1)/2, sp.S(5)/2, -sp.S(1)/2,
                    2, 2, 1, sp.S(1)/2, -1, 2, -2, -2, sp.S(3)/2, -2]

    # If 'term' is given, use only the specified term instead of the entire operator
    if term is not None:
        terms, coefficients = [terms[term - 1]], [coefficients[term - 1]]

    curvature = sum(c * t for c, t in zip(coefficients, terms))

    return curvature


def one_vertex(expr):
    '''
    Given an expression representing an operator acting on a generic reduced spin
    network state, generates the corresponding expression for a one-vertex state
    '''

    # Identify all nodes with v
    subs_v = [(x, v['0']) for x in v.values()]

    # Identify all edges e['x...'] with e_x, and the same for e_y and e_z
    subs_e = []
    ex, ey, ez = sp.symbols('e_x e_y e_z')
    for k, v_ in e.items():
        if k[0] == 'x':
            subs_e.append((v_, ex))
        elif k[0] == 'y':
            subs_e.append((v_, ey))
        elif k[0] == 'z':
            subs_e.append((v_, ez))

    # Substitute the reduced flux operators with their eigenvalues
    j_x, j_y, j_z = sp.symbols('j_x j_y j_z', positive=True)
    subs_p = [(p_x(v['0']), j_x), (p_y(v['0']), j_y), (p_z(v['0']), j_z)]

    return sp.expand(expr.subs(subs_v).subs(subs_e).subs(subs_p))


def print_terms(expr, latex=False):
    '''
    Prints the given expression, one term per line. If 'latex=True' is passed, the
    output is formatted as LaTeX code.

    The terms are grouped according to which of the labels x, y and z appear in them,
    and the terms within each group are sorted roughly from 'simplest' to 'most
    complicated'.
    '''

    # s(e)^2 + c(e)^2 = 1
    expr = sp.expand(expr)
    expr = expr.subs([(s(e_)**2, 1 - c(e_)**2) for e_ in e.values()])
    expr = sp.simplify(expr)

    terms = sp.expand(expr).as_ordered_terms()
    terms.sort(key=lambda x: (len(x.atoms(sp.Function)), x.as_coeff_mul()[0]))

    grouped_terms = defaultdict(list)

    for t in terms:
        labels = ''.join(char for char in 'xyz' if char in repr(t))
        grouped_terms[labels].append(t)

    for k, v in grouped_terms.items():
        if not latex:
            for t in v:
                print(t)
        else:
            print(f"%%% Terms with {', '.join(k)}", '\n')
            for t in v:
                print(r'\begin{equation}')
                print(sp.latex(t))
                print(r'\end{equation}', '\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-v', '--one-vertex', action='store_true',
                        help='Generate an expression representing the action of the '
                        'operator on a one-vertex state. (By default the action on a '
                        'generic reduced spin network state is given)')
    parser.add_argument('-l', '--latex', action='store_true',
                        help='Format the output as LaTeX code')
    parser.add_argument('-t', '--term', type=int,
                        help='Instead of the entire operator, consider only a single '
                        'term (1-16) in Eq. (3.2)')
    args = parser.parse_args()

    # Global variables representing the operators p_a(v), c(e), s(e) and V(v)
    # 
    # Treating the operators as commutative is equivalent to a factor ordering where
    # the flux operators are ordered to the right of the holonomy operators, after the
    # flux operators are replaced with their eigenvalues
    p_x, p_y, p_z = sp.symbols('p_x p_y p_z', cls=sp.Function, commutative=True)
    c, s = sp.symbols('c s', cls=sp.Function, commutative=True)
    V = sp.symbols('V', cls=sp.Function, commutative=True)

    _1, I = sp.S(1), sp.I   # noqa: E741 - shorthand for symbolic 1 and imaginary unit

    v = nodes()
    e = edges()

    R = curvature(args.term)

    if args.one_vertex:
        R = one_vertex(R)

    print_terms(R, latex=args.latex)
