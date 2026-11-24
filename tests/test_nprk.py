import numpy as np

from math import isclose

import pytest
from firedrake import *
from irksome import Dt, MeshConstant, NonlinearlyPartitionedRKTimeStepper, IMIMNPRKTableau, IMEXNPRKTableau, NPRKTableau


@pytest.mark.parametrize("stage_weight, expected", [(1.0, True), (0.999999, False)])
def test_nprk_stiff_accuracy(stage_weight, expected):
    tableau = NPRKTableau({(1, 1, 0): stage_weight}, {(1, 0): 1.0}, order=1)
    assert tableau.is_stiffly_accurate is expected


def burgers(nprk_tableau, order, N):
    msh = UnitIntervalMesh(N)
    V = FunctionSpace(msh, "CG", order)
    MC = MeshConstant(msh)
    dt = MC.Constant(0.1 / N)
    t = MC.Constant(0.0)
    (x,) = SpatialCoordinate(msh)

    # Choose uexact so rhs is nonzero
    uexact = cos(pi*x)*exp(-t)
    EPS = 1/200.0
    rhs = Dt(uexact) + uexact*uexact.dx(0) - EPS*div(grad(uexact))
    u = Function(V)
    u.interpolate(uexact)

    u1, u2 = Function(V), Function(V)

    v = TestFunction(V)

    F = (
        inner(Dt(u), v) * dx
        + inner(u2*u1.dx(0), v) * dx
        + inner(EPS*grad(u1), grad(v)) * dx
        - inner(rhs, v) * dx
    )

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}

    stepper = NonlinearlyPartitionedRKTimeStepper(F, nprk_tableau, t, dt, u, u1, u2,
                                                  arg1_solver_parameters=luparams,
                                                  arg2_solver_parameters=luparams,
                                                  mass_parameters=luparams)

    t_end = 0.1
    while float(t) < t_end:
        if float(t) + float(dt) > t_end:
            dt.assign(t_end - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))

    return (errornorm(uexact, u) / norm(uexact))


@pytest.mark.parametrize("nprk_t", [
    pytest.param(IMIMNPRKTableau("mp"), id="Midpoint"),
    pytest.param(IMIMNPRKTableau("cn"), id="Crank-Nicholson"),
    pytest.param(IMIMNPRKTableau("stiff2"), id="Stiff-2"),
    pytest.param(IMEXNPRKTableau("euler"), id="IMEX Forward Euler"),
    pytest.param(IMEXNPRKTableau("2[32]a"), id="IMEX 2[32]a"),
    pytest.param(IMEXNPRKTableau("2[32]b"), id="IMEX 2[32]b"),
])
def test_1d_burgers(nprk_t):
    order = nprk_t.order
    errs = np.array([burgers(nprk_t, order, 10*2**p) for p in [3, 4]])
    print(errs)
    conv = np.log2(errs[0]/errs[1])
    print(conv)
    assert conv > order-0.4


def heat_dirichletbc(nprk_tableau):
    # Boundary values
    u_0 = Constant(2.0)
    u_1 = Constant(3.0)

    N = 10
    x0 = 0.0
    x1 = 10.0
    msh = IntervalMesh(N, x1)
    V = FunctionSpace(msh, "CG", 1)
    MC = MeshConstant(msh)
    dt = MC.Constant(1.0 / N)
    t = MC.Constant(0.0)
    (x,) = SpatialCoordinate(msh)

    # Method of manufactured solutions copied from Heat equation demo.
    S = Constant(2.0)
    C = Constant(1000.0)
    B = (x - Constant(x0)) * (x - Constant(x1)) / C
    R = (x * x) ** 0.5
    # Note end linear contribution
    uexact = (
        B * atan(t) * (pi / 2.0 - atan(S * (R - t)))
        + u_0
        + ((x - x0) / x1) * (u_1 - u_0)
    )
    rhs = Dt(uexact) - div(grad(uexact))
    u = Function(V)
    u.interpolate(uexact)

    u1, u2 = Function(V), Function(V)

    v = TestFunction(V)
    F = (
        inner(Dt(u), v) * dx
        + 0.5*inner(grad(u1), grad(v)) * dx
        + 0.5*inner(grad(u2), grad(v)) * dx
        - inner(rhs, v) * dx
    )

    bc = [
        DirichletBC(V, u_1, 2),
        DirichletBC(V, u_0, 1),
    ]

    luparams = {"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}

    stepper = NonlinearlyPartitionedRKTimeStepper(F, nprk_tableau, t, dt, u, u1, u2, bcs=bc,
                                                  arg1_solver_parameters=luparams,
                                                  arg2_solver_parameters=luparams,
                                                  mass_parameters=luparams)

    t_end = 2.0
    while float(t) < t_end:
        print("Current time: ", float(t))
        if float(t) + float(dt) > t_end:
            dt.assign(t_end - float(t))
        stepper.advance()
        t.assign(float(t) + float(dt))
        # Check solution and boundary values
        assert errornorm(uexact, u) / norm(uexact) < 10.0 ** -3
        assert isclose(u.at(x0), u_0)
        assert isclose(u.at(x1), u_1)


@pytest.mark.parametrize("nprk_t", [IMIMNPRKTableau("cn"),
                                    IMIMNPRKTableau("stiff2")],
                         ids=["Crank-Nicholson", "2nd Order Stiffly Accurate"])
def test_1d_heat_dirichletbc(nprk_t):
    heat_dirichletbc(nprk_t)
