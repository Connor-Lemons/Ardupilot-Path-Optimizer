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
        tolerance (float, default=0.0): Required spatial tolerance, must be 0 [m]
        phi(float): Bearing in radians [rad]
    '''
    def __init__(self, x: float, y: float, z: float, phi_deg: float, **kwargs):
        if "directional" in kwargs:
            raise ValueError(
                "Cannot set 'directional' for StartConstraint; defaults to True."
            )
        if "tolerance" in kwargs:
            raise ValueError(
                "Cannot set 'tolerance' for StartConstraint: defaults to 0.0."
            )
        super().__init__(x=x, y=y, z=z, phi_deg=phi_deg, directional=True, tolerance=0.0, **kwargs)

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
        radius (float, optional, default=None): Required turn radius, negative for CCW, positive for CW [m]
        radius_tol (float, optional, default=None): Tolerance on radius constraint, set to None to disable [m]
        speed (float, optional, default=None): Required speed [m/s]
        speed_tol (float, optional, default=None): Tolerance on speed constraint, set to None to disable [m/s]
        climb (float, optional, default=None): Required climb rate [m/s]
        climb_tol (float, optional, default=-None): Tolerance on climb constraint, set to None to disable [m/s]
        duration (float, optional, default=None): Duration before spatial point that the constraints will be met [s]

    Properties:
        phi(float): Bearing in radians [rad]
    '''
    radius: Optional[float] = None
    radius_tol: Optional[float] = None
    speed: Optional[float] = None
    speed_tol: Optional[float] = None
    climb: Optional[float] = None
    climb_tol: Optional[float] = None
    duration: Optional[float] = None

    def __post_init__(self):
        if self.radius_tol is not None:
            if self.radius is None:
                raise ConstraintError("radius_tol is set, but radius is not.")
            if self.radius_tol < 0:
                raise ValueError("radius_tol must be nonnegative.")
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
        
    def trajectory(self):
        '''
        Returns the full trajectory as a list of constraints.
        '''
        return [self.start, *self.constraints]
    
    def trajectory_states(self, full_size=True):
        '''
        Returns the full trajectory as a list of states (np arrays). If False is passed, only returns spatial coordinates of states.
        '''
        return np.array([self.start.state(full_size=full_size), *(constraint.state(full_size=full_size) for constraint in self.constraints)])
    
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
    time_weight: Optional[float] = 1
    turn_weight: Optional[float] = 1
    speed_weight: Optional[float] = 1
    climb_weight: Optional[float] = 1
    segments: Optional[int] = 10
    points: Optional[int] = 10
    tol: Optional[float] = 1e-8

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
    r_inv_sol: List[float]
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
        self.problem = None
        self.solution = None
        self.Sol = None
        self.fig = None
        self.__post_init__()

    def __post_init__(self):
        for constraint in self.traj.constraints:
            if isinstance(constraint, ExtendedConstraint):
                if constraint.radius is not None and (np.abs(constraint.radius) < self.ap.min_turn or np.abs(constraint.radius) > self.ap.max_turn):
                    warnings.warn(f"Radius constraint |{constraint.radius}| [m] is out of given Ardupilot bounds [{self.ap.min_turn}, {self.ap.max_turn}]. "
                                  "This is likely to cause errors with the optimizer.", ConstraintWarning)
                if constraint.speed is not None and (constraint.speed < self.ap.V_min or constraint.speed > self.ap.V_max):
                    warnings.warn(f"Speed constraint {constraint.speed} [m/s] is out of given Ardupilot bounds [{self.ap.V_min}, {self.ap.V_max}]. "
                                  "This is likely to cause errors with the optimizer.", ConstraintWarning)
                if constraint.climb is not None and (constraint.climb < -self.ap.max_desc or constraint.climb > self.ap.max_climb):
                    warnings.warn(f"Climb constraint {constraint.climb} [m/s] is out of given Ardupilot bounds [-{self.ap.max_desc}, {self.ap.max_climb}]. "
                                  "This is likely to cause errors with the optimizer.", ConstraintWarning)

    def setup(self):
        '''
        Creates the YAPSS problem to solve. Can be precomputed to improve speed of self.solve().
        '''
        traj = self.traj
        ap = self.ap
        optim_params = self.optim_params

        W_t, W_r, W_v, W_c = optim_params.get_weights()

        num_constr = len(traj.constraints)
        tot_phase = num_constr*traj.phases_per_constraint
        last_phase_idx = tot_phase-1

        problem = yapss.Problem(name = "Constraints to Trajectory",
                                nx = [4]*tot_phase,
                                ns = 3*tot_phase,
                                nd = 5*(tot_phase-1))
        
        def objective(arg):
            param = arg.parameter
            objective = W_t*arg.phase[last_phase_idx].final_time**2
            for p in range(tot_phase):
                objective += (
                W_r*param[3*p]**2 +
                W_v*(param[3*p+1] - ap.V_cruise)**2 +
                W_c*param[3*p+2]**2
                )
            arg.objective = objective

        def continuous(arg):
            params = arg.parameter
            for p in range(tot_phase):
                x, y, z, phi = arg.phase[p].state
                r_inv, V, omega = params[3*p:3*p+3]
                xdot = V*math.cos(phi)
                ydot = V*math.sin(phi)
                zdot = omega
                phidot = V*r_inv
                arg.phase[p].dynamics = [xdot, ydot, zdot, phidot]

        def discrete(arg):
            discrete = []
            for p in range(last_phase_idx):
                discrete.append(arg.phase[p].final_time - arg.phase[p+1].initial_time)
                discrete.extend(arg.phase[p].final_state - arg.phase[p+1].initial_state)
            arg.discrete = discrete

        problem.functions.objective = objective
        problem.functions.continuous = continuous
        problem.functions.discrete = discrete

        for p in range(tot_phase):
            problem.bounds.phase[p].initial_time.lower = 0
            problem.bounds.parameter.lower[3*p] = -1/ap.min_turn
            problem.bounds.parameter.upper[3*p] = 1/ap.min_turn
            problem.bounds.parameter.lower[3*p+1] = ap.V_min
            problem.bounds.parameter.upper[3*p+1] = ap.V_max
            problem.bounds.parameter.lower[3*p+2] = -ap.max_desc
            problem.bounds.parameter.upper[3*p+2] = ap.max_climb
            problem.guess.phase[p].time = [0, 1]
        problem.bounds.discrete.lower[:] = problem.bounds.discrete.upper[:] = 0

        problem.bounds.phase[0].initial_time.upper = 0
        problem.bounds.phase[0].initial_state.lower = problem.bounds.phase[0].initial_state.upper = traj.start.state()
        for i in range(num_constr):
            current_phase_idx = (i+1)*traj.phases_per_constraint-1
            current_phase = problem.bounds.phase[current_phase_idx]
            current_constraint = traj.constraints[i]
            current_phase.final_state.lower[0:3] = current_constraint.state(False) - np.array([current_constraint.tolerance]*3)
            current_phase.final_state.upper[0:3] = current_constraint.state(False) + np.array([current_constraint.tolerance]*3)
            if current_constraint.directional:
                current_phase.final_state.lower[3] = current_phase.final_state.upper[3] = current_constraint.phi

            if isinstance(current_constraint, ExtendedConstraint):
                if current_constraint.radius_tol is not None:
                    radius_bounds = [1/(current_constraint.radius - current_constraint.radius_tol), 1/(current_constraint.radius + current_constraint.radius_tol)]
                    problem.bounds.parameter.lower[3*current_phase_idx] = min(radius_bounds)
                    problem.bounds.parameter.upper[3*current_phase_idx] = max(radius_bounds)
                if current_constraint.speed_tol is not None:
                    problem.bounds.parameter.lower[3*current_phase_idx+1] = current_constraint.speed - current_constraint.speed_tol
                    problem.bounds.parameter.upper[3*current_phase_idx+1] = current_constraint.speed + current_constraint.speed_tol
                if current_constraint.climb_tol is not None:
                    climb_bounds = [current_constraint.climb - current_constraint.climb_tol, current_constraint.climb + current_constraint.climb_tol]
                    problem.bounds.parameter.lower[3*current_phase_idx+2] = min(climb_bounds)
                    problem.bounds.parameter.upper[3*current_phase_idx+2] = max(climb_bounds)
                if current_constraint.duration is not None:
                    current_phase.duration.lower = current_phase.duration.upper = current_constraint.duration

        problem.mesh.phase[0].collocation_points = optim_params.segments * [optim_params.points]
        problem.mesh.phase[0].fraction = optim_params.segments * [1 / optim_params.segments]
        problem.ipopt_options.tol = optim_params.tol

        problem.ipopt_options.print_user_options = "no"
        problem.ipopt_options.print_level = 0

        self.problem = problem
        return self.problem

    def solve(self):
        '''
        Solves the YAPSS problem and returns the YAPSS solution object. If the problem has not been setup, it will be computed before solving.
        '''
        if self.problem is None:
            self.setup()
        self.solution = self.problem.solve()
        return self.solution

    def extract(self):
        '''
        Extracts the data from the YAPSS solution object and stitches the phase data together. The resulting Solution object is nicer to work with for most cases, though the Solution object intentionally omits some data from the YAPSS solution object (most notably, optimizer information and solution statistics).
        '''
        if self.solution is None:
            self.solve()
        sol = self.solution
        traj = self.traj
        ap = self.ap

        num_constr = len(traj.constraints)
        tot_phase = num_constr*traj.phases_per_constraint
        last_phase_idx = tot_phase-1

        t_sol = []
        t_phase_sol = []
        x_sol = []
        y_sol = []
        z_sol = []
        phi_sol = []
        phi_deg_sol = []
        r_inv_sol = []
        r_sol = []
        V_sol = []
        omega_sol = []

        for p in range(tot_phase):
            t_temp = sol.phase[p].time
            t_phase_sol.append(t_temp[0])
            if p == last_phase_idx:
                t_phase_sol.append(t_temp[-1])
            x_temp, y_temp, z_temp, phi_temp = sol.phase[p].state
            t_sol.extend(t_temp)
            x_sol.extend(x_temp)
            y_sol.extend(y_temp)
            z_sol.extend(z_temp)
            phi_sol.extend(phi_temp)
            phi_deg_sol.extend(np.rad2deg(phi_temp))
            r_inv_temp, V_temp, omega_temp = sol.parameter[3*p:3*p+3]

            if np.abs(r_inv_temp) < np.reciprocal(ap.max_turn):
                r_inv_temp = 0
            r_inv_sol.extend([r_inv_temp]*2)
            if r_inv_temp == 0:
                r_sol.extend([r_inv_temp]*2)
            else:
                r_sol.extend([np.reciprocal(r_inv_temp)]*2)

            V_sol.extend([V_temp]*2)
            omega_sol.extend([omega_temp]*2)

        self.Sol = TrajectorySolution(t_sol, t_phase_sol, x_sol, y_sol, z_sol, phi_sol, phi_deg_sol, r_inv_sol, r_sol, V_sol, omega_sol)
        return self.Sol

    def plot(self):
        '''
        Generates relevant plots of the trajectory.
        '''
        if self.Sol is None:
            self.extract()
        traj = self.traj
        ap = self.ap
        sol = self.solution
        Sol = self.Sol

        num_constr = len(traj.constraints)
        tot_phase = num_constr*traj.phases_per_constraint
        last_phase_idx = tot_phase-1

        fig = plt.figure(figsize=(20,16))
        fig.tight_layout()

        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        for p in range(tot_phase):
            x_temp, y_temp, z_temp, phi_temp = sol.phase[p].state
            ax1.plot(x_temp, y_temp, z_temp)
        ax1.scatter(*traj.start.state(False), color='green')
        for i in range(num_constr):
            ax1.scatter(*traj.constraints[i].state(False), color='red')
        ax1.set_aspect('equal', adjustable='datalim')
        ax1.set_xlabel("x [m]")
        ax1.set_ylabel("y [m]")
        ax1.set_zlabel("z [m]")
        ax1.set_title("Trajectory")
        ax1.grid(True)

        ax2 = fig.add_subplot(2, 2, 2)
        for p in range(tot_phase):
            t_temp = sol.phase[p].time
            r_inv_temp, V_temp, omega_temp = sol.parameter[3*p:3*p+3]
            if np.abs(r_inv_temp) < np.reciprocal(ap.max_turn):
                r_inv_temp = 0
            if r_inv_temp == 0:
                r_temp = r_inv_temp
            else:
                r_temp = np.reciprocal(r_inv_temp)
            ax2.plot([t_temp[0], t_temp[-1]], [r_temp, r_temp])
        ax2.set_xlim(Sol.t_sol[0], Sol.t_sol[-1])
        ax2.set_xticks(Sol.t_phase_sol)
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Turn Radius [m]")
        ax2.set_title("Turn Radius Parameters")
        ax2.grid(True)

        ax3 = fig.add_subplot(2, 2, 3)
        for p in range(tot_phase):
            t_temp = sol.phase[p].time
            r_inv_temp, V_temp, omega_temp = sol.parameter[3*p:3*p+3]
            ax3.plot([t_temp[0], t_temp[-1]], [V_temp, V_temp])
        ax3.set_xlim(Sol.t_sol[0], Sol.t_sol[-1])
        ax3.set_xticks(Sol.t_phase_sol)
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("Velocity [m/s]")
        ax3.set_title("Velocity Parameters")
        ax3.grid(True)

        ax4 = fig.add_subplot(2, 2, 4)
        for p in range(tot_phase):
            t_temp = sol.phase[p].time
            r_inv_temp, V_temp, omega_temp = sol.parameter[3*p:3*p+3]
            ax4.plot([t_temp[0], t_temp[-1]], [omega_temp, omega_temp])
        ax4.set_xlim(Sol.t_sol[0], Sol.t_sol[-1])
        ax4.set_xticks(Sol.t_phase_sol)
        ax4.set_xlabel("Time [s]")
        ax4.set_ylabel("Climb Rate [m/s]")
        ax4.set_title("Altitude Parameters")
        ax4.grid(True)

        plt.close(fig)

        self.fig = fig
        return self.fig