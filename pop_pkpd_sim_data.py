import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import math, random


def generate_population_parameters_with_covariates(n_individuals, base_dict, covariates_dict):
    """
    Generate individual population parameters influenced by covariates with small variability.
    
    Parameters:
    - n_individuals (int): Number of individuals
    - base_dict (dict): Dictionary of base parameters, covariate influences, and noise levels
    - covariates_dict (dict): Dictionary of covariates values for each individual
    
    Returns:
    - population_data (dict): Dictionary where each key is an individual's ID and each value
                              is a dictionary of parameters and covariate values for that individual
    """
    population_data = {}

    # Generate parameters for each individual
    for ind in range(n_individuals):
        individual_data = {}
        
        # Store covariate values for this individual
        individual_covs = {cov: values[ind] for cov, values in covariates_dict.items()}
        
        # Calculate parameters for this individual based on covariate effects
        individual_params = {}
        for param, param_info in base_dict.items():
            base_value = param_info['base']
            noise = param_info['noice']
            cov_influence = param_info['cov']
            
            # Start with the base parameter value
            param_value = base_value
            
            # Apply each covariate's influence on the parameter
            for cov_name, cov_details in cov_influence.items():
                cov_value = individual_covs[cov_name]
                cov_fn = cov_details['cov_fn']
                args = cov_details['args']
                
                if cov_fn is not None:
                    param_value *= cov_fn(cov_value, **args)
            
            # Add noise for variability
            param_value += np.random.normal(0, noise)
            individual_params[param] = param_value

        # Combine parameters and covariates for this individual
        individual_data['params'] = individual_params
        individual_data['covariates'] = individual_covs
        population_data[ind] = individual_data

    return population_data

def linear_cov(cov, slope):
    return cov * slope

def power_cov(cov, exp, base=0):
    return (cov/base) ** exp

def transform_population_data(population_data):
    """
    Transforms population data into a dictionary with lists for each covariate and parameter.
    
    Parameters:
    - population_data (dict): Dictionary where each individual's data is stored under an ID key
    
    Returns:
    - transformed_data (dict): Dictionary where each covariate and parameter has a list of values for all individuals
    """
    transformed_data = {'covariates': {}, 'params': {}}
    
    # Initialize lists for each covariate and parameter based on the first individual's data
    first_individual = population_data[0]
    for cov in first_individual['covariates'].keys():
        transformed_data['covariates'][cov] = []
    for param in first_individual['params'].keys():
        transformed_data['params'][param] = []
    
    # Fill in the lists with values from each individual
    for ind_data in population_data.values():
        for cov, value in ind_data['covariates'].items():
            transformed_data['covariates'][cov].append(value)
        for param, value in ind_data['params'].items():
            transformed_data['params'][param].append(value)
    
    return transformed_data

def plot_param_vs_covariate(ax, x, y, x_label, y_label, title):
    sns.scatterplot(x=x, y=y, ax=ax)
    sns.regplot(x=x, y=y, ax=ax, scatter=False, color="red", line_kws={"linestyle": "dashed"})
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

def pk_model(t, y, KaA1, KeA1, KaA2, KeA2, k1, k2):
    A1a, A1b, A2a, A2b = y
    dA1a_dt = -KaA1 * A1a - k1 * A1a  
    dA2a_dt = -KaA2 * A2a + k1 * A1a  
    dA1b_dt = KaA1 * A1a - KeA1 * A1b - k2 * A1b
    dA2b_dt = KaA2 * A2a - KeA2 * A2b + k2 * A1b
    return [dA1a_dt, dA1b_dt, dA2a_dt, dA2b_dt]

def generate_dosing_schedule(dose, time_scale, time_scale_frequency, num_doses):
    """
    Generate a dosing schedule.

    Parameters:
    - dose (float): Dose amount in mg.
    - time_scale (float): Base time interval (e.g., 24 hours).
    - time_scale_frequency (int): Frequency of time scales between doses.
    - num_doses (int): Number of doses to include in the schedule.

    Returns:
    - List of tuples [(time, dose), ...] where time is the dosing time and dose is the dose amount.
    """
    dosing_schedule = [(time_scale * time_scale_frequency * i, dose) for i in range(num_doses)]
    return dosing_schedule

def simulate_dosing_schedule(dose, time_scale, time_scale_frequency, num_doses, person_id, population_params):
    """
    Simulates the pharmacokinetics model for a single individual with a specific dosing schedule.

    Parameters:
    - dose (float): Dose amount in mg.
    - time_scale (float): Base time interval (e.g., 24 hours).
    - time_scale_frequency (int): Frequency of doses (e.g., 3 for once every 3 time scales).
    - num_doses (int): Number of doses to simulate.
    - person_id (int): Unique identifier for the individual.

    Returns:
    - DataFrame with the simulation results for the given individual.
    """
    dosing_schedule = generate_dosing_schedule(dose, time_scale, time_scale_frequency, num_doses)
    initial_conditions = [0, 0, 0, 0]  
    t_eval_per_interval = 10
    results = []

    # Solve for each interval between doses
    for i in range(len(dosing_schedule) - 1):
        t_start, dose = dosing_schedule[i]
        t_end = dosing_schedule[i + 1][0]
        
        initial_conditions[0] = dose  
        t_eval_interval = np.linspace(t_start, t_end, t_eval_per_interval)
        
        solution = solve_ivp(
            pk_model, [t_start, t_end], initial_conditions, t_eval=t_eval_interval, method='RK45', 
            args=(population_params['KaA1'][person_id-1], population_params['KeA1'][person_id-1], population_params['KaA2'][person_id-1], population_params['KeA2'][person_id-1], population_params['k1'][person_id-1], population_params['k2'][person_id-1])
        )
        
        # Calculate "Time Since Last Dose" for this interval
        time_since_last_dose = solution.t - t_start
        
        interval_df = pd.DataFrame({
            'Time': solution.t,
            'Time Since Last Dose': time_since_last_dose,
            'A1a': solution.y[0],
            'A1b': solution.y[1],
            'A2a': solution.y[2],
            'A2b': solution.y[3],
            'Dose': [dose if math.isclose(t, t_start, abs_tol=0.1) else 0 for t in solution.t],
            'Dosing Cycle': i + 1,
            'DSFQ' : [time_scale_frequency for _ in range(time_since_last_dose.__len__())],
            'Person ID': person_id,
            'dosing_amount' : dose
        })
        results.append(interval_df)
        
        initial_conditions = [solution.y[0][-1], solution.y[1][-1], solution.y[2][-1], solution.y[3][-1]]
    
    return pd.concat(results, ignore_index=True)

def generate_individuals(num_individuals, population_params, dose_choices=[100], time_scale_choices=[24], time_scale_frequency=[2,3], num_doses = [4, 6]):
    individuals = []
    for i in range(num_individuals):
        dose = random.choice(dose_choices)
        time_scale = random.choice(time_scale_choices)
        if isinstance(time_scale_frequency, list):         
            time_scale_frequency_ = random.randint(time_scale_frequency[0], time_scale_frequency[1])
        else: time_scale_frequency_ = time_scale_frequency
        if isinstance(num_doses, list):         
            num_doses = random.randint(num_doses[0], num_doses[1])
        else: num_doses_ = num_doses
        person_id = i + 1

        individual = {
            'dose': dose,
            'time_scale': time_scale,
            'time_scale_frequency': time_scale_frequency_, #Note this is implemented as such that the dosing time is time_scare * time_scale_frequency 
            'num_doses': num_doses_,
            'person_id': person_id,
            'population_params' : population_params
        }
        individuals.append(individual)

    return individuals

def remove_duplicate_times(df, time_column="Time", group_column="Person ID"):
    df = df.sort_values(by=[group_column, time_column])
    
    def keep_second_occurrence(group):
        duplicates = group.duplicated(subset=[time_column], keep=False)
        
        result = group.copy()
        
        count = {}
        for idx in result.index:
            time_value = result.at[idx, time_column]
            if duplicates[idx]:
                count[time_value] = count.get(time_value, 0) + 1
                # Keep the second occurrence 
                if count[time_value] == 2:
                    continue  
                else:
                    result.drop(index=idx, inplace=True)  
        return result

    df = df.groupby(group_column).apply(keep_second_occurrence)

    return df.reset_index(drop=True)