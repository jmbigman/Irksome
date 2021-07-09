from .getForm import getForm, AI, IAinv
from firedrake import NonlinearVariationalProblem as NLVP
from firedrake import NonlinearVariationalSolver as NLVS
from firedrake import Function, norm
import numpy


class TimeStepper:
    """Front-end class for advancing a time-dependent PDE via a Runge-Kutta
    method.

    :arg F: A :class:`ufl.Form` instance describing the semi-discrete problem
            F(t, u; v) == 0, where `u` is the unknown
            :class:`firedrake.Function and `v` is the
            :class:firedrake.TestFunction`.
    :arg butcher_tableau: A :class:`ButcherTableau` instance giving
            the Runge-Kutta method to be used for time marching.
    :arg t: A :class:`firedrake.Constant` instance that always
            contains the time value at the beginning of a time step
    :arg dt: A :class:`firedrake.Constant` containing the size of the
            current time step.  The user may adjust this value between
            time steps, but see :class:`AdaptiveTimeStepper` for a
            method that attempts to do this automatically.
    :arg u0: A :class:`firedrake.Function` containing the current
            state of the problem to be solved.
    :arg bcs: An iterable of :class:`firedrake.DirichletBC` containing
            the strongly-enforced boundary conditions.  Irksome will
            manipulate these to obtain boundary conditions for each
            stage of the RK method.
    :arg bc_type: How to manipulate the strongly-enforced boundary
            conditions to derive the stage boundary conditions.
            Should be a string, either "DAE", which implements BCs as
            constraints in the style of a differential-algebraic
            equation, or "ODE", which takes the time derivative of the
            boundary data and evaluates this for the stage values
    :arg solver_parameters: A :class:`dict` of solver parameters that
            will be used in solving the algebraic problem associated
            with each time step.

    """
    def __init__(self, F, butcher_tableau, t, dt, u0, bcs=None,
                 solver_parameters=None, bc_type="DAE", splitting=AI):
        self.splitting=splitting
        self.u0 = u0
        self.t = t
        self.dt = dt
        self.num_fields = len(u0.function_space())
        self.num_stages = len(butcher_tableau.b)
        self.butcher_tableau = butcher_tableau

        bigF, stages, bigBCs, bigBCdata = \
            getForm(F, butcher_tableau, t, dt, u0, bcs, bc_type, splitting)

        self.stages = stages
        self.bigBCs = bigBCs
        self.bigBCdata = bigBCdata
        problem = NLVP(bigF, stages, bigBCs)
        self.solver = NLVS(problem, solver_parameters=solver_parameters)

        if self.num_stages == 1 and self.num_fields == 1:
            self.ks = (stages,)
        else:
            self.ks = stages.split()

    def _update(self):
        """Assuming the algebraic problem for the RK stages has been
        solved, updates the solution.  This will not typically be
        called by an end user."""
        b = self.butcher_tableau.b
        dtc = float(self.dt)
        u0 = self.u0
        ns = self.num_stages
        nf = self.num_fields

        # FIXME: Lift this outside of the update
        A1, A2 = self.splitting(self.butcher_tableau.A)
        b = numpy.linalg.solve(A2, b)
        
        ks = self.ks
        for s in range(ns):
            for i, u0d in enumerate(u0.dat):
                u0d.data[:] += dtc * b[s] * ks[nf*s+i].dat.data_ro

    def advance(self):
        """Advances the system from time `t` to time `t + dt`.
        Note: overwrites the value `u0`."""
        for gdat, gcur, gmethod in self.bigBCdata:
            gmethod(gcur, self.u0)

        self.solver.solve()

        self._update()


class AdaptiveTimeStepper(TimeStepper):
    """Front-end class for advancing a time-dependent PDE via a Runge-Kutta
    method.

    :arg F: A :class:`ufl.Form` instance describing the semi-discrete problem
            F(t, u; v) == 0, where `u` is the unknown
            :class:`firedrake.Function and `v` is the
            :class:firedrake.TestFunction`.
    :arg butcher_tableau: A :class:`ButcherTableau` instance giving
            the Runge-Kutta method to be used for time marching.
    :arg t: A :class:`firedrake.Constant` instance that always
            contains the time value at the beginning of a time step
    :arg dt: A :class:`firedrake.Constant` containing the size of the
            current time step.  The user may adjust this value between
            time steps, but see :class:`AdaptiveTimeStepper` for a
            method that attempts to do this automatically.
    :arg u0: A :class:`firedrake.Function` containing the current
            state of the problem to be solved.
    :arg tol: The temporal ttruncation error tolerance
    :arg dtmin: Minimal acceptable time step.  An exception is raised
            if the step size drops below this threshhold.
    :arg bcs: An iterable of :class:`firedrake.DirichletBC` containing
            the strongly-enforced boundary conditions.  Irksome will
            manipulate these to obtain boundary conditions for each
            stage of the RK method.
    :arg solver_parameters: A :class:`dict` of solver parameters that
            will be used in solving the algebraic problem associated
            with each time step.
    """
    def __init__(self, F, butcher_tableau, t, dt, u0,
                 tol=1.e-6, dtmin=1.e-5, bcs=None, solver_parameters=None,
                 bc_type="DAE"):
        assert butcher_tableau.btilde is not None
        super(AdaptiveTimeStepper, self).__init__(F, butcher_tableau,
                                                  t, dt, u0, bcs,
                                                  solver_parameters, bc_type)
        self.tol = tol
        self.dt_min = dtmin
        self.delb = butcher_tableau.b - butcher_tableau.btilde
        self.error_func = Function(u0.function_space())

    def _estimate_error(self):
        """Assuming that the RK stages have been evaluated, estimates
        the temporal truncation error by taking the norm of the
        difference between the new solutions computed by the two
        methods.  Typically will not be called by the end user."""
        dtc = float(self.dt)
        delb = self.delb

        ks = self.ks
        nf = self.num_fields
        for e in self.error_func.dat:
            e.data[:] = 0.0
        for s in range(self.num_stages):
            for i, e in enumerate(self.error_func.dat):
                e.data[:] += dtc * delb[i] * ks[nf*s+i].dat.data_ro
        return norm(self.error_func)

    def advance(self):
        """Attempts to advances the system from time `t` to time `t +
        dt`.  If the error threshhold is exceeded, will adaptively
        decrease the time step until the step is accepted.  Also
        predicts new time step once the step is accepted.

        Note: overwrites the value `u0`."""
        print("\tTrying dt=", float(self.dt))
        while 1:
            for gdat, gcur, gmethod in self.bigBCdata:
                gmethod(gcur, self.u0)

            self.solver.solve()
            err = self._estimate_error()

            print("\tTruncation error", err)
            q = 0.84 * (self.tol / err)**(1./(self.butcher_tableau.order-1))
            print("\tq factor:", q)
            if q <= 0.1:
                q = 0.1
            elif q >= 4.0:
                q = 4.0

            dtnew = q * float(self.dt)

            if err >= self.tol:
                print("\tShrinking time step to ", dtnew)
                self.dt.assign(dtnew)
            elif dtnew <= self.dt_min:
                raise RuntimeError("Minimum time step threshold violated")
            else:
                print("\tStep accepted, new time step is ", dtnew)
                self._update()
                self.dt.assign(dtnew)
                return (err, dtnew)
