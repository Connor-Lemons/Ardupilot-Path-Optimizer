from dataclasses import dataclass, field
from typing import Optional, List
import warnings
import numpy as np
import matplotlib.pyplot as plt
import yapss
from yapss import math

@dataclass
class Constraint:
    '''
    Defines a generic spatial constraint for the optimizer.
    
    A constraint is defined by a position in space, and can optionally include a required bearing.

    Attributes:
        x (float): X position of constraint [m]
        y (float): Y position of constraint [m]
        z (float): Z position of constraint [m]
        phi_deg (float, optional, default=0.0): Required bearing [deg]
        directional (bool, optional, default=False): Whether to enforce the bearing
        tolerance (float, optional, default=1.0): Required spatial tolerance, must be >= 1.0 [m]

    Properties:
        phi(float): Bearing in radians [rad]
    '''
    x: float
    y: float
    z: float
    phi_deg: Optional[float] = 0.0
    directional: Optional[bool] = False
    tolerance: Optional[float] = 1.0

    @property
    def phi(self):
        return np.deg2rad(self.phi_deg)

    def __post_init__(self):
        if self.tolerance < 1.0 and not isinstance(self, StartConstraint):
            raise ValueError("tolerance must be >= 1.0.")

    def state(self, full_size=True):
        '''
        Returns the coordinates and bearing of the constraint. If False is passed, only returns spatial coordinates. Note that the bearing is in radians.
        '''
        if full_size == False:
            return np.array([self.x, self.y, self.z])
        else:
            return np.array([self.x, self.y, self.z, self.phi])

@dataclass
class StartConstraint(Constraint):
    '''
    Defines the start constraint for the optimizer.
    
    A start constraint is defined by a position in space, and must include a required bearing.

    Attributes:
        x (float): X position of constraint [m]
        y (float): Y position of constraint [m]
        z (float): Z position of constraint [m]
        phi_deg (float): Required bearing [deg]

    Properties:
        directional (bool, default=True): Whether to enforce the bearing, must be True
        tolerance (float, default=1.0): Required spatial tolerance, must be 1 [m]
    '''
    def __init__(self, x: float, y: float, z: float, phi_deg: float, **kwargs):
        if "directional" in kwargs:
            raise ValueError(
                "Cannot set 'directional' for StartConstraint; defaults to True."
            )
        if "tolerance" in kwargs:
            raise ValueError(
                "Cannot set 'tolerance' for StartConstraint: defaults to 1.0."
            )
        super().__init__(x=x, y=y, z=z, phi_deg=phi_deg, directional=True, tolerance=1.0, **kwargs)

@dataclass
class ExtendedConstraint(Constraint):
    '''
    Defines a more advanced constraint which optionally includes speed and bank angle constraints.

    An extended constraint has all the attributes of a normal constraint, but allows for constraining the speed and/or bank angle of the aircraft when entering the constraint. Note that the use of extended constraints will likely require more phases per constraint to handle.

    The corresponding tolerance must be defined for the given constraint, otherwise it won't be active.

    Attributes:
        x (float): X position of constraint [m]
        y (float): Y position of constraint [m]
        z (float): Z position of constraint [m]
        phi_deg (float, optional, default=0.0): Required bearing [deg]
        directional (bool, optional, default=False): Whether to enforce the bearing
        tolerance (float, optional, default=1.0): Required spatial tolerance, must be >= 1.0 [m]
        bank_deg (float, optional, default=None): Required bank angle [deg]
        bank_deg_tol (float, optional, default=None): Tolerance on bank constraint, set to None to disable [deg]
        speed (float, optional, default=None): Required speed [m/s]
        speed_tol (float, optional, default=None): Tolerance on speed constraint, set to None to disable [m/s]
        climb (float, optional, default=None): Required climb rate [m/s]
        climb_tol (float, optional, default=-None): Tolerance on climb constraint, set to None to disable [m/s]
        duration (float, optional, default=None): Duration before spatial point that the constraints will be met [s]

    Properties:
        phi (float): Bearing in radians [rad]
        bank (float): Bank angle in radians [rad]
    '''
    bank_deg: Optional[float] = None
    bank_deg_tol: Optional[float] = None
    speed: Optional[float] = None
    speed_tol: Optional[float] = None
    climb: Optional[float] = None
    climb_tol: Optional[float] = None
    duration: Optional[float] = None

    @property
    def bank(self):
        if self.bank_deg is None:
            return None
        return np.deg2rad(self.bank_deg)

    @property
    def bank_tol(self):
        if self.bank_deg_tol is None:
            return None
        return np.deg2rad(self.bank_deg_tol)

    def __post_init__(self):
        if self.bank_deg_tol is not None:
            if self.bank_deg is None:
                raise ConstraintError("bank_deg_tol is set, but bank_deg is not.")
            if self.bank_deg_tol < 0:
                raise ValueError("bank_deg_tol must be nonnegative.")
        if self.speed_tol is not None:
            if self.speed is None:
                raise ConstraintError("speed_tol is set, but speed is not.")
            if self.speed_tol < 0:
                raise ValueError("speed_tol must be nonnegative.")
        if self.climb_tol is not None:
            if self.climb is None:
                raise ConstraintError("climb_tol is set, but climb is not.")
            if self.climb_tol < 0:
                raise ValueError("climb_tol must be nonnegative.")
        if self.speed is not None and self.speed < 0:
            raise ValueError("speed must be nonnegative.")
        if self.duration is not None:
            if self.duration < 0:
                raise ValueError("duration must be nonnegative.")
            if self.duration == 0:
                warnings.warn(f"Current duration of {self.duration} [s] may lead to optional parameters being ignored by the optimizer. Consider increasing the duration.", ConstraintWarning)
        return super().__post_init__()

@dataclass
class Trajectory:
    '''
    A container for multiple constraints which together define a full trajectory.

    The StartConstraint must be provided, is handled separately, and should not be included in the list of Constraints.

    Attributes:
        start (StartConstraint): Defines the start of the trajectory
        constraints (List[Constraint]): Defines the constraints of the trajectory in the order given
        phases_per_constraint (int, optional, default=3): Number of optimizer phases allowed per constraint
    '''
    start: StartConstraint
    constraints: List[Constraint] = field(default_factory=list)
    phases_per_constraint: Optional[int] = 3

    def __post_init__(self):
        if not isinstance(self.start, StartConstraint):
            raise TypeError("First Constraint must be a StartConstraint.")
        if not isinstance(self.constraints, list):
            raise TypeError("Trajectory must be a list of Constraint objects.")
        if len(self.constraints) == 0:
            raise ValueError("Trajectory must contain at least one constraint other than the StartConstraint.")
        for constraint in self.constraints:
            if isinstance(constraint, StartConstraint):
                raise TypeError("StartConstraints are not permitted in the constraint list.")
        
    def __len__(self):
        return len(self.constraints) + 1
        
    def add_constraint(self, constraint:Constraint):
        '''
        Adds a constraint to the end of trajectory.
        '''
        if isinstance(constraint, StartConstraint):
            raise TypeError("Can only add Constraint objects, not StartConstraint objects")
        self.constraints.append(constraint)

    def insert_constraint(self, constraint:Constraint, idx=1):
        '''
        Adds a constraint to the specified index of the trajectory. Indexing includes the start constraint, so setting idx to 1 will put the constraint between the start constraint and the first normal constraint.
        '''
        if isinstance(constraint, StartConstraint):
            raise TypeError("Can only insert Constraint objects, not StartConstraint objects")
        if idx == 0:
            raise ValueError("Cannot insert constraint before start.")
        self.constraints.insert(idx-1, constraint)

    def sub_constraint(self):
        '''
        Removes a constraint from the end of the trajectory.
        '''
        if len(self.constraints) == 1:
            raise ValueError("Must have at least one constraint to define the trajectory.")
        self.constraints.pop()

    def remove_constraint(self, idx=1):
        '''
        Removes a constraint from the specified index of the trajectory. Indexing includes the start constraint, so setting idx to 1 will remove the constraint just after the start constraint.
        '''
        if idx == 0:
            raise ValueError("Cannot remove start constraint.")
        if len(self.constraints) == 1:
            raise ValueError("Must have at least one constraint to define the trajectory.")
        self.constraints.pop(idx-1)
        
    def trajectory(self) -> list[Constraint]:
        '''
        Returns the full trajectory as a list of constraints.
        '''
        return [self.start, *self.constraints]
    
    def trajectory_states(self, full_size=True):
        '''
        Returns the full trajectory as a list of states (np arrays). If False is passed, only returns spatial coordinates of states.
        '''
        return np.array([self.start.state(full_size=full_size), *(constraint.state(full_size=full_size) for constraint in self.constraints)])
    
    def distances(self):
        '''
        Calculates the Euclidian distances between each constraint.
        '''
        dist = []
        states = self.trajectory_states(False)
        for p in range(len(self.constraints)):
            dist.append(np.linalg.norm(states[p+1] - states[p]))
        return np.array(dist)
    
@dataclass
class ArdupilotParameters:
    '''
    Contains relevant Ardupilot parameters for the optimizer to reference.

    Values can be found in Mission Planner's Full Parameter List. Default values are from the default plane in Mission Planner's SITL.

    Attributes:
        V_max (float): AIRSPEED_MAX [m/s]
        V_min (float): AIRSPEED_MIN [m/s]
        V_cruise (float): AIRSPEED_CRUISE [m/s]
        max_climb (float): TECS_CLIMB_MAX [m/s]
        max_desc (float): TECS_SINK_MAX [m/s]
        roll_limit (float): ROLL_LIMIT_DEG [deg]
        roll_min (float): LEVEL_ROLL_LIMIT [deg]

    Properties:
        roll_min_rad (float): roll_min in radians [rad]
        roll_limit_rad (float): roll_limit in radians [rad]
        max_turn (float): Maximum turn radius [m]
        min_turn (float): Minimum turn radius [m]
    '''
    V_max: Optional[float] = 30
    V_min: Optional[float] = 10
    V_cruise: Optional[float] = 22
    max_climb: Optional[float] = 5
    max_desc: Optional[float] = 5
    roll_limit: Optional[float] = 65
    roll_min: Optional[float] = 5

    @property
    def roll_min_rad(self):
        return np.deg2rad(self.roll_min)
    
    @property
    def roll_limit_rad(self):
        return np.deg2rad(self.roll_limit)
    
    @property
    def max_turn(self):
        return self.V_cruise**2/(9.80665*math.tan(self.roll_min_rad))
    
    @property
    def min_turn(self):
        return self.V_cruise**2/(9.80665*math.tan(self.roll_limit_rad))
    
@dataclass
class OptimizerParameters:
    '''
    Contains optimizer parameters.

    Attributes:
        time_weight (float, optional, default=1.0): Weighting on final time
        turn_weight (float, optional, default=1.0): Weighting on turn radius
        speed_weight (float, optional, default=1.0): Weighting on velocity deviation from cruise
        climb_weight (float, optional, default=1.0): Weighting on requested altitude change
        segments (int, optional, default=1): Number of segments in the mesh
        points (int, optional, default=1): Number of collocation points per segment in the mesh
        tol (float, optional, default=1e-8): Ipopt tolerance
    '''
    time_weight: Optional[float] = 10
    turn_weight: Optional[float] = 1
    speed_weight: Optional[float] = 10
    climb_weight: Optional[float] = 1
    segments: Optional[int] = 9
    points: Optional[int] = 3
    tol: Optional[float] = 1e-6

    def get_weights(self):
        return np.array([self.time_weight,
                         self.turn_weight,
                         self.speed_weight,
                         self.climb_weight])
    
@dataclass(frozen=True)
class TrajectorySolution:
    '''
    Contains trajectory solution.

    Attributes:
        t_sol (List[float]): Vector of times [s]
        t_phase_sol (List[float]): Vector of start/end times for each phase [s]
        x_sol (List[float]): Vector of X positions [m]
        y_sol (List[float]): Vector of Y positions [m]
        z_sol (List[float]): Vector of Z positions [m]
        phi_sol (List[float]): Vector of bearings [rad]
        phi_deg_sol(List[float]): Vector of bearings in degrees [deg]
        r_inv_sol (List[float]): Vector of reciprocal turn radii [1/m]
        r_sol (List[float]): Vector of turn radii [m]
        V_sol (List[float]): Vector of velocities [m/s]
        omega_sol (List[float]): Vector of climb rates [m/s]
    '''
    t_sol: List[float]
    t_phase_sol: List[float]
    x_sol: List[float]
    y_sol: List[float]
    z_sol: List[float]
    phi_sol: List[float]
    phi_deg_sol: List[float]
    bank_sol: List[float]
    r_sol: List[float]
    V_sol: List[float]
    omega_sol: List[float]

class ConstraintWarning(UserWarning):
    '''
    Creates a ConstraintWarning for easy user filtering.
    '''
    pass

class ConstraintError(ValueError):
    '''
    Creates a ConstraintError for easy user filtering.
    '''
    pass

class Optimizer:
    '''
    Defines the optimizer, which includes the required inputs, the problem, the solution, and the Solution object.

    Attributes:
        traj (Trajectory): Trajectory object to solve
        ap (ArdupilotParameters): Required Ardupilot parameters
        optim_params (OptimizerParameters): Optimizer settings
    '''
    def __init__(self, traj:Trajectory, ap:ArdupilotParameters, optim_params: OptimizerParameters):
        self.traj = traj
        self.ap = ap
        self.optim_params = optim_params
        self.initial = None
        self.fig_initial = None
        self.problem = None
        self.solution = None
        self.Sol = None
        self.fig = None
        self.__post_init__()

    def __post_init__(self):
        for constraint in self.traj.constraints:
            if isinstance(constraint, ExtendedConstraint):
                if constraint.bank_deg_tol is not None and (np.abs(constraint.bank_deg) < self.ap.roll_min or np.abs(constraint.bank_deg) > self.ap.roll_limit):
                    warnings.warn(f"Bank constraint |{constraint.bank_deg}| [deg] is out of given Ardupilot bounds [{self.ap.roll_min}, {self.ap.roll_limit}]. "
                                  "This is likely to cause errors with the optimizer.", ConstraintWarning)
                if constraint.speed_tol is not None and (constraint.speed < self.ap.V_min or constraint.speed > self.ap.V_max):
                    warnings.warn(f"Speed constraint {constraint.speed} [m/s] is out of given Ardupilot bounds [{self.ap.V_min}, {self.ap.V_max}]. "
                                  "This is likely to cause errors with the optimizer.", ConstraintWarning)
                if constraint.climb_tol is not None and (constraint.climb < -self.ap.max_desc or constraint.climb > self.ap.max_climb):
                    warnings.warn(f"Climb constraint {constraint.climb} [m/s] is out of given Ardupilot bounds [{-self.ap.max_desc}, {self.ap.max_climb}]. "
                                  "This is likely to cause errors with the optimizer.", ConstraintWarning)
                    
    def initial_solver(self, problem, st, tgt, dir):
        '''
        Solver for the initial guess.
        '''
        dist = np.sqrt((st[0] - tgt[0])**2 + (st[1] - tgt[1])**2 + (st[2] - tgt[2])**2)
        z_dist = np.abs(st[2] - tgt[2])

        def objective(arg):
            x, y, z, phi= arg.phase[0].final_state
            arg.objective = arg.phase[0].final_time + (x - tgt[0])**2 + (y - tgt[1])**2 + (z - tgt[2])**2

        problem.functions.objective = objective

        problem.bounds.phase[0].final_time.lower = min(dist/self.ap.V_cruise, 2*np.pi*self.ap.max_turn/self.ap.V_cruise + z_dist/self.ap.V_cruise)
        problem.bounds.phase[0].final_time.upper = 2*np.pi*self.ap.max_turn/self.ap.V_cruise + z_dist/max(self.ap.max_desc, self.ap.max_climb)
        problem.bounds.phase[0].initial_state.lower = problem.bounds.phase[0].initial_state.upper = np.concatenate((st, [dir]))

        problem.guess.phase[0].state = [
            [st[0], tgt[0]],
            [st[1], tgt[1]],
            [st[2], tgt[2]],
            [dir, 0]
            ]

        if np.cos(dir)*(tgt[1] - st[1]) - np.sin(dir)*(tgt[0] - st[0]) > 0:
            problem.bounds.parameter.lower = [self.ap.roll_min_rad, -self.ap.max_desc]
            problem.bounds.parameter.upper = [self.ap.roll_limit_rad, self.ap.max_climb]
            problem.guess.parameter = [np.pi/4, 0]
        else:
            problem.bounds.parameter.lower = [-self.ap.roll_limit_rad, -self.ap.max_desc]
            problem.bounds.parameter.upper = [-self.ap.roll_min_rad, self.ap.max_climb]
            problem.guess.parameter = [-np.pi/4, 0]

        sol = problem.solve()
        t = sol.phase[0].time
        x, y, z, dir = sol.phase[0].state
        bank, climb = sol.parameter

        return t, np.array([x, y, z, dir]), np.array([bank, self.ap.V_cruise, climb])
    
    def presolve(self):
        '''
        Presolves a relaxed version of the problem to generate an initial guess.
        '''
        self.initial = {}

        prob = yapss.Problem(name = "Constraints to Trajectory",
                                nx = [4],
                                ns = 2)

        def continuous(arg):
            x, y, z, phi = arg.phase[0].state
            bank, omega = arg.parameter
            xdot = self.ap.V_cruise*math.cos(phi)
            ydot = self.ap.V_cruise*math.sin(phi)
            zdot = omega
            phidot = 9.80665*math.tan(bank)/self.ap.V_cruise
            arg.phase[0].dynamics = [xdot, ydot, zdot, phidot]

        prob.functions.continuous = continuous

        prob.bounds.phase[0].initial_time.lower = prob.bounds.phase[0].initial_time.upper = 0
        prob.guess.phase[0].time = [0, 10]

        segments, points = 9, 3
        prob.mesh.phase[0].collocation_points = segments * [points]
        prob.mesh.phase[0].fraction = segments * [1 / segments]
        prob.ipopt_options.tol = 1e-3

        constr_list = self.traj.trajectory()
        for c in range(len(constr_list)-1):
            start = constr_list[c]
            end = constr_list[c+1]

            if isinstance(start, StartConstraint):
                dir1 = start.phi
            else:
                dir1 = np.atan2((start.y - constr_list[c-1].y), (start.x - constr_list[c-1].x))

            if np.atan2((end.y - start.y), (end.x - start.x)) == dir1:
                t_end = np.linalg.norm(end.state(False) - start.state(False))/self.ap.V_cruise
                self.initial[f"constr{c}_time"] = np.array([0.0, t_end])
                self.initial[f"constr{c}_state"] = np.array([[start.x, end.x], [start.y, end.y], [start.z, end.z], [dir1, dir1]])
                self.initial[f"constr{c}_params"] = np.array([0, self.ap.V_cruise, 0])
            else:
                self.initial[f"constr{c}_time"], self.initial[f"constr{c}_state"], self.initial[f"constr{c}_params"] = self.initial_solver(prob, start.state(False), end.state(False), dir1)

        return self.initial
    
    def presolve_plot(self):
        '''
        Plots the guess trajectory.
        '''
        # Checks to see if extract() has been run and runs it if not.
        if self.initial is None:
            self.presolve()
        traj = self.traj
        ap = self.ap
        initial = self.initial

        # Helpful reference numbers for cleaner indexing later
        num_constr = len(traj.constraints)

        # Initialize figure.
        fig = plt.figure(figsize=(20,16))
        fig.tight_layout()

        # Plot the trajectory.
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        # Go phase by phase so that each phase gets its own color.
        for c in range(num_constr):
            x_temp, y_temp, z_temp, phi_temp = initial[f"constr{c}_state"]
            ax1.plot(x_temp, y_temp, z_temp)
        for i in range(num_constr):
            ax1.scatter(*traj.constraints[i].state(False), color='red', label='Constraint')
        ax1.scatter(*traj.start.state(False), color='green', label='Start Constraint')
        ax1.set_aspect('equal', adjustable='datalim')
        ax1.set_xlabel("x [m]")
        ax1.set_ylabel("y [m]")
        ax1.set_zlabel("z [m]")
        ax1.set_title("Trajectory")
        ax1.grid(True)

        # Plot the turn radius.
        ax2 = fig.add_subplot(2, 2, 2)
        # Plot the speed.
        ax3 = fig.add_subplot(2, 2, 3)
        # Plot the climb rate.
        ax4 = fig.add_subplot(2, 2, 4)

        t_phase_temp = []
        t_phase_temp.append(initial[f"constr0_time"][0])
        for c in range(num_constr):
            t_temp = initial[f"constr{c}_time"]
            t_phase_temp.append(t_temp[-1])
            bank_temp, V_temp, omega_temp = initial[f"constr{c}_params"]
            if np.abs(bank_temp) < ap.roll_min_rad:
                r_temp = 0
            else:
                r_temp = V_temp**2/(9.80665*np.tan(bank_temp))
            ax2.plot([t_temp[0], t_temp[-1]], [r_temp, r_temp])
            ax3.plot([t_temp[0], t_temp[-1]], [V_temp, V_temp])
            ax4.plot([t_temp[0], t_temp[-1]], [omega_temp, omega_temp])

        ax2.set_xlim(t_phase_temp[0], t_phase_temp[-1])
        ax2.set_xticks(t_phase_temp)
        ax3.set_xlim(t_phase_temp[0], t_phase_temp[-1])
        ax3.set_xticks(t_phase_temp)
        ax4.set_xlim(t_phase_temp[0], t_phase_temp[-1])
        ax4.set_xticks(t_phase_temp)

        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Turn Radius [m]")
        ax2.set_title("Turn Radius Parameters")
        ax2.grid(True)
        
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("Speed [m/s]")
        ax3.set_title("Speed Parameters")
        ax3.grid(True)

        ax4.set_xlabel("Time [s]")
        ax4.set_ylabel("Climb Rate [m/s]")
        ax4.set_title("Altitude Parameters")
        ax4.grid(True)

        plt.close(fig)

        # Return the figure.
        self.fig = fig
        return self.fig

    def setup(self):
        '''
        Creates the YAPSS problem to solve. Can be precomputed to improve speed of self.solve().
        '''
        # Checks to see if presolve() has been run and runs it if not.
        if self.initial is None:
            self.presolve()

        traj = self.traj
        ap = self.ap
        optim_params = self.optim_params

        # Extract the weights to form the cost function to minimize.
        W_t, W_r, W_v, W_c = optim_params.get_weights()

        # Helpful reference numbers for cleaner indexing later
        num_constr = len(traj.constraints)
        tot_phase = num_constr*traj.phases_per_constraint
        last_phase_idx = tot_phase-1

        #
        constr_dist = traj.distances()

        # Creates the YAPSS problem, nx defines the number of states (and phases via its structure), ns defines the number of parameters (global to the problem), and nd defines the number of discrete constraints necessary.
        problem = yapss.Problem(name = "Constraints to Trajectory",
                                nx = [4]*tot_phase,
                                ns = 3*tot_phase,
                                nd = 5*(tot_phase-1),
                                nq = [1]*tot_phase)
        
        # Add in guesses from preliminary solutions
        global_t0 = 0.0
        for c in range(num_constr):
            time_leg = np.asarray(self.initial[f"constr{c}_time"]).copy()
            state_leg = np.asarray(self.initial[f"constr{c}_state"]).copy()
            state_leg[3] = np.unwrap(state_leg[3])
            param_leg = np.asarray(self.initial[f"constr{c}_params"]).copy()

            time_leg = time_leg - time_leg[0]
            time_leg, unique_idx = np.unique(time_leg, return_index=True)
            state_leg = state_leg[:, unique_idx]
            t_inter = np.linspace(time_leg[0], time_leg[-1], traj.phases_per_constraint + 1)

            for i in range(traj.phases_per_constraint):
                t0 = t_inter[i]
                t1 = t_inter[i+1]
                t_range_idx = (time_leg > t0 + 1e-12) & (time_leg < t1 - 1e-12)
                t_range = np.concatenate(([t0], time_leg[t_range_idx], [t1]))
                t_range = np.unique(t_range)
                state_sub = np.vstack([np.interp(t_range, time_leg, state_leg[k]) for k in range(4)])
                t_sub = t_range + global_t0
                param_sub = param_leg

                phase_idx = i + c*traj.phases_per_constraint
                problem.guess.phase[phase_idx].time = t_sub
                problem.guess.phase[phase_idx].state = state_sub
                problem.guess.parameter[3*phase_idx:3*phase_idx+3] = param_sub

            global_t0 += time_leg[-1]

        t_guess = global_t0
        
        # Define the objective function.
        def objective(arg):
            arg.objective = W_t*arg.phase[-1].final_time/t_guess
            for p in range(tot_phase):
                arg.objective += arg.phase[p].integral[0]

        # Define the continuous function (system dynamics and controls).
        def continuous(arg):
            params = arg.parameter
            # Define the system dynamics for each phase.
            for p in range(tot_phase):
                x, y, z, phi = arg.phase[p].state
                bank, V, omega = params[3*p:3*p+3]
                xdot = V*math.cos(phi)
                ydot = V*math.sin(phi)
                zdot = omega
                phidot = 9.80665*math.tan(bank)/V
                arg.phase[p].dynamics = [xdot, ydot, zdot, phidot]
                # Define the integral cost
                arg.phase[p].integrand[0] = (
                    W_r*(bank/ap.roll_limit_rad)**2
                    + W_v*((V - ap.V_cruise)/ap.V_cruise)**2
                    + W_c*(omega/ap.max_climb)**2
                )

        # Define the discrete function (discrete constraints).
        def discrete(arg):
            discrete = []
            # Make sure state and time of previous phase match state and time of next phase.
            for p in range(last_phase_idx):
                discrete.append(arg.phase[p].final_time - arg.phase[p+1].initial_time)
                discrete.extend(arg.phase[p].final_state - arg.phase[p+1].initial_state)
            arg.discrete = discrete

        # Pass the defined functions to YAPSS.
        problem.functions.objective = objective
        problem.functions.continuous = continuous
        problem.functions.discrete = discrete

        # Preliminarily bound the controls for each phase based on Ardupilot limits. ExtendedConstraints can override these later.
        for p in range(tot_phase):
            problem.bounds.phase[p].initial_time.lower = 0
            problem.bounds.parameter.lower[3*p] = -ap.roll_limit_rad
            problem.bounds.parameter.upper[3*p] = ap.roll_limit_rad
            problem.bounds.parameter.lower[3*p+1] = ap.V_min
            problem.bounds.parameter.upper[3*p+1] = ap.V_max
            problem.bounds.parameter.lower[3*p+2] = -ap.max_desc
            problem.bounds.parameter.upper[3*p+2] = ap.max_climb

        # Make sure that all discrete constraints are enforced.
        problem.bounds.discrete.lower[0:5*(tot_phase-1)] = -1e-8
        problem.bounds.discrete.upper[0:5*(tot_phase-1)] = 1e-8
        problem.bounds.discrete.lower[0:5*(tot_phase-1):5] = 0
        problem.bounds.discrete.upper[0:5*(tot_phase-1):5] = 0
        

        # Define initial conditions based on StartConstraint of Trajectory.
        problem.bounds.phase[0].initial_time.upper = 0
        problem.bounds.phase[0].initial_state.lower[0:3] = traj.start.state(False) - np.array([traj.start.tolerance]*3)
        problem.bounds.phase[0].initial_state.upper[0:3] = traj.start.state(False) + np.array([traj.start.tolerance]*3)
        problem.bounds.phase[0].initial_state.lower[3] = traj.start.phi
        problem.bounds.phase[0].initial_state.upper[3] = traj.start.phi

        # Iteratively enforce constraints at correct phases to maintain phases_per_constraint.
        for i in range(num_constr):
            current_phase_idx = (i+1)*traj.phases_per_constraint-1
            current_phase = problem.bounds.phase[current_phase_idx]
            current_constraint = traj.constraints[i]
            current_phase.final_state.lower[0:3] = current_constraint.state(False) - np.array([current_constraint.tolerance]*3)
            current_phase.final_state.upper[0:3] = current_constraint.state(False) + np.array([current_constraint.tolerance]*3)
            # Enforce bearing if Constraint is directional.
            if current_constraint.directional:
                current_phase.final_state.lower[3] = current_constraint.phi
                current_phase.final_state.upper[3] = current_constraint.phi

            # Enforce ExtendedConstraint if applicable.
            if isinstance(current_constraint, ExtendedConstraint):
                # Turn radius constraint.
                if current_constraint.bank_tol is not None:
                    bank_bounds = [current_constraint.bank - current_constraint.bank_tol, current_constraint.bank + current_constraint.bank_tol]
                    problem.bounds.parameter.lower[3*current_phase_idx] = min(bank_bounds)
                    problem.bounds.parameter.upper[3*current_phase_idx] = max(bank_bounds)
                # Speed constraint.
                if current_constraint.speed_tol is not None:
                    problem.bounds.parameter.lower[3*current_phase_idx+1] = current_constraint.speed - current_constraint.speed_tol
                    problem.bounds.parameter.upper[3*current_phase_idx+1] = current_constraint.speed + current_constraint.speed_tol
                # Climb constraint.
                if current_constraint.climb_tol is not None:
                    climb_bounds = [current_constraint.climb - current_constraint.climb_tol, current_constraint.climb + current_constraint.climb_tol]
                    problem.bounds.parameter.lower[3*current_phase_idx+2] = min(climb_bounds)
                    problem.bounds.parameter.upper[3*current_phase_idx+2] = max(climb_bounds)
                # Phase duration constraint.
                if current_constraint.duration is not None:
                    current_phase.duration.lower = current_phase.duration.upper = current_constraint.duration

        # Set the problem mesh.
        for p in range(tot_phase):
            problem.mesh.phase[p].collocation_points = optim_params.segments * [optim_params.points]
            problem.mesh.phase[p].fraction = optim_params.segments * [1 / optim_params.segments]
        # Set the IPOPT tolerance.
        problem.ipopt_options.tol = optim_params.tol
        # Suppress IPOPT outputs.
        problem.ipopt_options.print_user_options = "no"
        problem.ipopt_options.print_level = 0

        # Create the problem
        self.problem = problem
        return self.problem

    def solve(self):
        '''
        Solves the YAPSS problem and returns the YAPSS solution object. If the problem has not been setup, it will be computed before solving.
        '''
        # Checks to see if setup() has been run and runs it if not.
        if self.problem is None:
            self.setup()

        # Solve the problem
        self.solution = self.problem.solve()
        return self.solution

    def extract(self):
        '''
        Extracts the data from the YAPSS solution object and stitches the phase data together. The resulting Solution object is nicer to work with for most cases, though the Solution object intentionally omits some data from the YAPSS solution object (most notably, optimizer information and solution statistics).
        '''
        # Checks to see if solve() has been run and runs it if not.
        if self.solution is None:
            self.solve()
        # Simplify variables.
        sol = self.solution
        traj = self.traj
        ap = self.ap

        # Define useful indices.
        num_constr = len(traj.constraints)
        tot_phase = num_constr*traj.phases_per_constraint
        last_phase_idx = tot_phase-1

        # Initialize lists to contain the solution.
        t_sol = []
        t_phase_sol = []
        x_sol = []
        y_sol = []
        z_sol = []
        phi_sol = []
        phi_deg_sol = []
        bank_sol = []
        r_sol = []
        V_sol = []
        omega_sol = []

        # Iterate throgh each phase and add the data to the matching solution array.
        for p in range(tot_phase):
            # Store the time endpoints of the phase separately for later reference.
            t_temp = sol.phase[p].time
            t_phase_sol.append(t_temp[0])
            if p == last_phase_idx:
                t_phase_sol.append(t_temp[-1])
            # Stitch the state together.
            x_temp, y_temp, z_temp, phi_temp = sol.phase[p].state
            t_sol.extend(t_temp)
            x_sol.extend(x_temp)
            y_sol.extend(y_temp)
            z_sol.extend(z_temp)
            phi_sol.extend(phi_temp)
            phi_deg_sol.extend(np.rad2deg(phi_temp))
            # Stitch the controls together
            bank_temp, V_temp, omega_temp = sol.parameter[3*p:3*p+3]
            # If the requested control radius is larger than the maximum allowed turn, treat it as a straight trajectory.
            bank_sol.extend([bank_temp])
            # Generate the radius solution.
            if bank_temp == 0:
                r_sol.extend([0])
            else:
                r_sol.extend([V_temp**2/(9.80665*np.tan(bank_temp))])
            V_sol.extend([V_temp])
            omega_sol.extend([omega_temp])

        # Extract the problem into a TrajectorySolution object.
        self.Sol = TrajectorySolution(t_sol, t_phase_sol, x_sol, y_sol, z_sol, phi_sol, phi_deg_sol, bank_sol, r_sol, V_sol, omega_sol)
        return self.Sol

    def plot(self):
        '''
        Generates relevant plots of the trajectory.
        '''
        # Checks to see if extract() has been run and runs it if not.
        if self.Sol is None:
            self.extract()
        traj = self.traj
        ap = self.ap
        sol = self.solution
        Sol = self.Sol

        # Handy indices.
        num_constr = len(traj.constraints)
        tot_phase = num_constr*traj.phases_per_constraint
        last_phase_idx = tot_phase-1

        # Initialize figure.
        fig = plt.figure(figsize=(20,16))
        fig.tight_layout()

        # Plot the trajectory.
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        # Go phase by phase so that each phase gets its own color.
        for p in range(tot_phase):
            x_temp, y_temp, z_temp, phi_temp = sol.phase[p].state
            ax1.plot(x_temp, y_temp, z_temp)
        for i in range(num_constr):
            ax1.scatter(*traj.constraints[i].state(False), color='red', label='Constraint')
        ax1.scatter(*traj.start.state(False), color='green', label='Start Constraint')
        ax1.set_aspect('equal', adjustable='datalim')
        ax1.set_xlabel("x [m]")
        ax1.set_ylabel("y [m]")
        ax1.set_zlabel("z [m]")
        ax1.set_title("Trajectory")
        ax1.grid(True)

        # Plot the turn radius.
        ax2 = fig.add_subplot(2, 2, 2)
        for p in range(tot_phase):
            t_temp = sol.phase[p].time
            bank_temp, V_temp, omega_temp = sol.parameter[3*p:3*p+3]
            if np.abs(bank_temp) < ap.roll_min_rad:
                r_temp = 0
            else:
                r_temp = V_temp**2/(9.80665*np.tan(bank_temp))
            ax2.plot([t_temp[0], t_temp[-1]], [r_temp, r_temp])
        ax2.set_xlim(Sol.t_sol[0], Sol.t_sol[-1])
        ax2.set_xticks(Sol.t_phase_sol)
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Turn Radius [m]")
        ax2.set_title("Turn Radius Parameters")
        ax2.grid(True)

        # Plot the speed.
        ax3 = fig.add_subplot(2, 2, 3)
        for p in range(tot_phase):
            t_temp = sol.phase[p].time
            bank_temp, V_temp, omega_temp = sol.parameter[3*p:3*p+3]
            ax3.plot([t_temp[0], t_temp[-1]], [V_temp, V_temp])
        ax3.set_xlim(Sol.t_sol[0], Sol.t_sol[-1])
        ax3.set_xticks(Sol.t_phase_sol)
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("Speed [m/s]")
        ax3.set_title("Speed Parameters")
        ax3.grid(True)

        # Plot the climb rate.
        ax4 = fig.add_subplot(2, 2, 4)
        for p in range(tot_phase):
            t_temp = sol.phase[p].time
            bank_temp, V_temp, omega_temp = sol.parameter[3*p:3*p+3]
            ax4.plot([t_temp[0], t_temp[-1]], [omega_temp, omega_temp])
        ax4.set_xlim(Sol.t_sol[0], Sol.t_sol[-1])
        ax4.set_xticks(Sol.t_phase_sol)
        ax4.set_xlabel("Time [s]")
        ax4.set_ylabel("Climb Rate [m/s]")
        ax4.set_title("Altitude Parameters")
        ax4.grid(True)

        plt.close(fig)

        # Return the figure.
        self.fig = fig
        return self.fig