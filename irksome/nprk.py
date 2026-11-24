"""Classes to implement non-linearly partitioned Runge-Kutta (NPRK) schemes based on
Buvoli and Southworth."""
import numpy as np

from firedrake import Function, NonlinearVariationalProblem, NonlinearVariationalSolver

from .ufl.deriv import TimeDerivative, expand_time_derivatives

from .tools import replace
from .constant import MeshConstant
from .nprk_tableaux import NPRKTableau


def getFormsNPRK(F, u0, u1, u2, t, dt):
    """Generates three forms: one for implicit solves in each of the arguments
    and one for mass solves"""

    V = F.arguments()[0].function_space()
    msh = V.mesh()
    MC = MeshConstant(msh)

    space_check = (V == u0.function_space()
                   == u1.function_space()
                   == u2.function_space())

    if not space_check:
        raise ValueError("All three functions and "
                         "Form argument must be on the same function space")

    # Stage value, implicitly solved for
    u_stage = Function(V)
    # Term formed from summing previous stages
    u_tilde = Function(V)
    # Argument kept explicit in implicit solves
    u_exp = Function(V)
    # Implicit coefficient
    a = MC.Constant(0.0)
    # Abscissae
    c = MC.Constant(0.0)

    # (NOTE): Replacing both u0 and Dt(u0) places stage values where auxiliary variables are
    #         and stage value updates where time derivatives are

    repl_1 = {u0: u_stage, TimeDerivative(u0): (u_stage - u_tilde)/(dt * a),
              u1: u_stage, u2: u_exp, t: t + c * dt}
    repl_2 = {u0: u_stage, TimeDerivative(u0): (u_stage - u_tilde)/(dt * a),
              u1: u_exp, u2: u_stage, t: t + c * dt}

    F_arg1 = replace(F, repl_1)
    F_arg2 = replace(F, repl_2)

    # Solution of explicit mass solve
    k = Function(V)
    # Functions for first and second argument of explicit mass solve
    u_arg1, u_arg2 = Function(V), Function(V)

    # (NOTE): Replacing both u0 and Dt(u0) allows the value of the auxiliary variables to be
    #         determined while solving for the stage derivative of time-dependent variables

    repl_mass = {u0: k, TimeDerivative(u0): k, u1: u_arg1, u2: u_arg2, t: t + c * dt}
    F_mass = replace(F, repl_mass)

    return u_stage, u_tilde, u_exp, (a, c), F_arg1, F_arg2, k, u_arg1, u_arg2, F_mass


class NonlinearlyPartitionedRKTimeStepper:
    """Class for advancing a time-dependent PDE through a non-linearly partitioned
    Runge-Kutta (NPRK) method. This requires one form with three functions corresponding to
    the time derivatives/auxiliary terms, the first argument terms and the second argument terms.

    :arg F: A :class:`ufl.Form` instance describing partitioned operator F(t, u1, u2) == 0,
             where `u1` and `u2` denote the first and second arguments
    :arg nprk_tableau: A :class:`nprk_tableaux.NPRKTableau` instance giving the NPRK scheme
    :arg t: A :class:`MeshConstant` attribute referring to the current time
    :arg dt: A :class:`MeshConstant` attribute referring to the current time step
    :arg u0: A :class:`firedrake.Function` instance containing the current solution value.
             All :class:`irksome.deriv.TimeDerivative`s and auxiliary variables in `F` should
             be described in terms of `u0`
    :arg u1: A :class:`firedrake.Function` instance indicating the first argument in `F`
    :arg u2: A :class:`firedrake.Function` instance indicating the second argument in `F`
    :arg bcs: An iterable of :class:`firedrake.DirichletBC` containing the strongly-enforced
              boundary conditions. These are applied at each stage of the RK method.
    :arg arg1_solver_parameters: A :class:`dict` of solver parameters that will be used in the
                                 implicit solves in the first argument.
    :arg arg2_solver_parameters: A :class:`dict` of solver parameters that will be used in the
                                 implicit solves in the second argument.
    :arg mass_parameters: A :class:`dict` of solver parameters that will be used in the mass
                          solves when the operator must be evaluated on previous stages.
    :arg appctx: An optional :class:`dict` containing application context.
    :arg nullspace: An optional null space object.
    """

    def __init__(self, F, nprk_tableau: NPRKTableau, t, dt, u0, u1, u2, bcs=None,
                 arg1_solver_parameters=None, arg2_solver_parameters=None,
                 mass_parameters=None, appctx=None, nullspace=None):

        if bcs is None:
            bcs = []

        self.tableau = nprk_tableau

        self.dt = dt
        self.u = u0

        # preprocess time derivatives
        F = expand_time_derivatives(F, t=t, timedep_coeffs=(u0,))

        self.u_stage, self.u_tilde, self.u_exp, self.ac, F_arg1, F_arg2, \
            self.k, self.u_arg1, self.u_arg2, F_mass = getFormsNPRK(F, u0, u1, u2, t, dt)

        V = u0.function_space()
        # Operator evaluations for the pairs of stage arguments
        self.fs = {tuple(jk): Function(V) for jk in nprk_tableau.all_indices}
        # Stage values, needed for most general tableaux
        self.us = [Function(V) for _ in range(nprk_tableau.num_stages)]

        # BCs can be passed straight through to stage value solvers
        arg_probs = tuple(NonlinearVariationalProblem(F_, self.u_stage, bcs)
                          for F_ in (F_arg1, F_arg2))

        # Additional BCs for mass solvers
        bcs_mass = []
        for bc in bcs:
            bcs_mass.append(bc.reconstruct(g=0.0))

        mass_prob = NonlinearVariationalProblem(F_mass, self.k, bcs_mass)

        appctx_irksome = {"F": F,
                          "nprk_tableau": nprk_tableau,
                          "t": t,
                          "dt": dt,
                          "u0": u0,
                          "bcs": bcs,
                          "bc_type": "DAE",
                          "stage_type": "nprk",
                          "nullspace": nullspace}
        if appctx is None:
            appctx = appctx_irksome
        else:
            appctx = {**appctx, **appctx_irksome}

        # One solver corresponding to each implicit argument
        self.solvers = tuple(NonlinearVariationalSolver(arg_prb, appctx=appctx,
                                                        solver_parameters=arg_prms,
                                                        nullspace=nullspace)
                             for arg_prb, arg_prms in zip(arg_probs, (arg1_solver_parameters,
                                                                      arg2_solver_parameters)))

        self.mass_solver = NonlinearVariationalSolver(mass_prob,
                                                      solver_parameters=mass_parameters,
                                                      nullspace=nullspace)

        self.num_steps = 0

        # Number of non-linear and linear iterations in solves in each argument
        self.num_nonlinear_iterations = np.array([0 for _ in self.solvers], dtype=int)
        self.num_linear_iterations = np.array([0 for _ in self.solvers], dtype=int)

        self.num_mass_nonlinear_iterations = 0
        self.num_mass_linear_iterations = 0

        if self.tableau.is_stiffly_accurate:
            self._finalize = self._finalize_stiffly_accurate
        else:
            self._finalize = self._finalize_general

    def advance(self):
        """Takes a single time step"""

        ns = self.tableau.num_stages
        args = self.tableau.impl_args
        coeffs = self.tableau.impl_coeffs
        impl_indices = self.tableau.impl_indices
        a_list = self.tableau.a_list
        exp_dict = self.tableau.expl_solve_dict
        abscissae = self.tableau.c_values

        a, c = self.ac

        u_stage = self.u_stage
        u_tilde = self.u_tilde
        fs = self.fs

        dtc = float(self.dt)

        self.u_stage.assign(self.u)
        self.us[0].assign(self.u_stage)

        # Loop over remaining stages
        for i in range(1, ns):

            # Calculate explicit operator evaluations from previous stage
            for jk in exp_dict.get(i-1, []):
                j_, k_ = jk[0], jk[1]
                self.u_arg1.assign(self.us[j_])
                self.u_arg2.assign(self.us[k_])
                c.assign(abscissae[j_])
                self.mass_solver.solve()
                self.num_mass_nonlinear_iterations += self.mass_solver.snes.getIterationNumber()
                self.num_mass_linear_iterations += self.mass_solver.snes.getLinearSolveIterations()
                fs[tuple(jk)].assign(self.k)

            # Assign implicit coefficient and abscissae
            a.assign(coeffs[i-1])
            c.assign(abscissae[impl_indices[i - 1, 0]])
            # Assign previous stage
            self.u_exp.assign(u_stage)

            # Sum over previous stages
            u_tilde.assign(self.u)
            a_indices, a_values = a_list[i-1]
            for jk, a_ijk in zip(a_indices, a_values):
                f_ = fs[tuple(jk)]
                for u_t_bit, fbit in zip(u_tilde.subfunctions, f_.subfunctions):
                    u_t_bit += dtc * float(a_ijk) * fbit

            # Implicit solve
            self.solvers[args[i-1]].solve()
            self.num_nonlinear_iterations[args[i-1]] += self.solvers[args[i-1]].snes.getIterationNumber()
            self.num_linear_iterations[args[i-1]] += self.solvers[args[i-1]].snes.getLinearSolveIterations()

            # Store stage value
            self.us[i].assign(u_stage)

            # Store operator evaluation in dictionary
            f_ = fs[tuple(impl_indices[i-1])]
            f_.assign((u_stage - u_tilde)/(dtc * a))

        c.assign(0.0)

        self._finalize()
        self.num_steps += 1

    def _finalize_general(self):
        """Finalize by summing over stage derivatives"""
        b_indices = self.tableau.b_indices
        b_values = self.tableau.b_values
        u = self.u
        fs = self.fs
        dtc = float(self.dt)

        for jk, b_jk in zip(b_indices, b_values):
            f_ = fs[tuple(jk)]
            for ubit, fbit in zip(u.subfunctions, f_.subfunctions):
                ubit += dtc * float(b_jk) * fbit

    def _finalize_stiffly_accurate(self):
        """Finalize stiffly accurate case by assigning last stage"""
        self.u.assign(self.u_stage)

    def solver_stats(self):
        """Returns solver data"""
        return (self.num_steps,
                self.num_nonlinear_iterations,
                self.num_linear_iterations,
                self.num_mass_nonlinear_iterations,
                self.num_mass_linear_iterations)
