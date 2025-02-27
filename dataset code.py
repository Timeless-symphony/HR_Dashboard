import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

# Initialize Faker
fake = Faker('en_US')
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# Configuration
num_records = 8950

# States & Cities
states_cities = {
    'New York': ['New York City', 'Buffalo', 'Rochester'],
    'Virginia': ['Virginia Beach', 'Norfolk', 'Richmond'],
    'Florida': ['Miami', 'Orlando', 'Tampa'],
    'Illinois': ['Chicago', 'Aurora', 'Naperville'],
    'Pennsylvania': ['Philadelphia', 'Pittsburgh', 'Allentown'],
    'Ohio': ['Columbus', 'Cleveland', 'Cincinnati'],
    'North Carolina': ['Charlotte', 'Raleigh', 'Greensboro'],
    'Michigan': ['Detroit', 'Grand Rapids', 'Warren']
}
states = list(states_cities.keys())
state_prob = [0.7, 0.02, 0.01, 0.03, 0.05, 0.03, 0.05, 0.11]
assigned_states = np.random.choice(states, size=num_records, p=state_prob)
assigned_cities = [np.random.choice(states_cities[state]) for state in assigned_states]

# Departments & Jobtitles
departments = ['HR', 'IT', 'Sales', 'Marketing', 'Finance', 'Operations', 'Customer Service']
departments_prob = [0.02, 0.15, 0.21, 0.08, 0.05, 0.30, 0.19]
jobtitles = {
    'HR': ['HR Manager', 'HR Coordinator', 'Recruiter', 'HR Assistant'],
    'IT': ['IT Manager', 'Software Developer', 'System Administrator', 'IT Support Specialist'],
    'Sales': ['Sales Manager', 'Sales Consultant', 'Sales Specialist', 'Sales Representative'],
    'Marketing': ['Marketing Manager', 'SEO Specialist', 'Content Creator', 'Marketing Coordinator'],
    'Finance': ['Finance Manager', 'Accountant', 'Financial Analyst', 'Accounts Payable Specialist'],
    'Operations': ['Operations Manager', 'Operations Analyst', 'Logistics Coordinator', 'Inventory Specialist'],
    'Customer Service': ['Customer Service Manager', 'Customer Service Representative', 'Support Specialist', 'Help Desk Technician']
}
jobtitles_prob = {
    'HR': [0.03, 0.3, 0.47, 0.2],  # HR Manager, HR Coordinator, Recruiter, HR Assistant
    'IT': [0.02, 0.47, 0.2, 0.31],  # IT Manager, Software Developer, System Administrator, IT Support Specialist
    'Sales': [0.03, 0.25, 0.32, 0.4],  # Sales Manager, Sales Consultant, Sales Specialist, Sales Representative
    'Marketing': [0.04, 0.25, 0.41, 0.3],  # Marketing Manager, SEO Specialist, Content Creator, Marketing Coordinator
    'Finance': [0.03, 0.37, 0.4, 0.2],  # Finance Manager, Accountant, Financial Analyst, Accounts Payable Specialist
    'Operations': [0.02, 0.2, 0.4, 0.38],  # Operations Manager, Operations Analyst, Logistics Coordinator, Inventory Specialist
    'Customer Service': [0.04, 0.3, 0.38, 0.28]  # Customer Service Manager, Customer Service Representative, Support Specialist, Help Desk Technician
}

# Educations
educations = ['High School', "Bachelor", "Master", 'PhD']

education_mapping = {
    'HR Manager': ["Master", "PhD"],
    'HR Coordinator': ["Bachelor", "Master"],
    'Recruiter': ["High School", "Bachelor"],
    'HR Assistant': ["High School", "Bachelor"],
    'IT Manager': ["PhD", "Master"],
    'Software Developer': ["Bachelor", "Master"],
    'System Administrator': ["Bachelor", "Master"],
    'IT Support Specialist': ["High School", "Bachelor"],
    'Sales Manager': ["Master","PhD"],
    'Sales Consultant': ["Bachelor", "Master", "PhD"],
    'Sales Specialist': ["Bachelor", "Master", "PhD"],
    'Sales Representative': ["Bachelor"],
    'Marketing Manager': ["Bachelor", "Master","PhD"],
    'SEO Specialist': ["High School", "Bachelor"],
    'Content Creator': ["High School", "Bachelor"],
    'Marketing Coordinator': ["Bachelor"],
    'Finance Manager': ["Master", "PhD"],
    'Accountant': ["Bachelor"],
    'Financial Analyst': ["Bachelor", "Master", "PhD"],
    'Accounts Payable Specialist': ["Bachelor"],
    'Operations Manager': ["Bachelor", "Master"],
    'Operations Analyst': ["Bachelor", "Master"],
    'Logistics Coordinator': ["Bachelor"],
    'Inventory Specialist': ["High School", "Bachelor"],
    'Customer Service Manager': ["Bachelor", "Master", "PhD"],
    'Customer Service Representative': ["High School", "Bachelor"],
    'Support Specialist': ["High School", "Bachelor"],
    'Customer Success Manager': ["Bachelor", "Master", "PhD"],
    'Help Desk Technician': ["High School", "Bachelor"]
}

# Hiring Date
# Define custom probability weights for each year
year_weights = {
    2015: 5,   # 15% probability
    2016: 8,   # 15% probability
    2017: 17,   # 20% probability
    2018: 9,  # 15% probability
    2019: 10,  # 10% probability
    2020: 11,  # 10% probability
    2021: 5,  # 8% probability
    2022: 12,  # 5% probability
    2023: 14,  # 2% probability
    2024: 9   # 2% probability
}

#This code defines a dictionary called year_weights in Python. This dictionary is designed to store and represent custom probability weights associated with different years. 
# These weights can be used in various applications where you need to randomly select a year, but not with uniform probability. Instead, you want some years to be more likely to be chosen than others.
'''
Breakdown

year_weights = { ... }: This line creates a dictionary and assigns it to the variable named year_weights. Dictionaries in Python are used to store key-value pairs.

2015: 5,, 2016: 8,, and so on: These are the key-value pairs within the dictionary.

Key: The year (e.g., 2015, 2016, 2017). The keys are integers in this case.

Value: The weight associated with that year (e.g., 5, 8, 17). The values are also integers. These weights represent the relative probability of selecting that year.
the numbers themselves (5, 8, 17, etc.) do not directly represent percentages. They are just relative weights. To get actual probabilities,
 you'd need to normalize these weights (i.e., divide each weight by the sum of all the weights).

 example:
def weighted_random_choice(weights):
    """
    Selects a random key from a dictionary based on its associated weight.

    Args:
        weights (dict): A dictionary where keys are the items to choose from,
                       and values are their corresponding weights.

    Returns:
        The randomly selected key.
    """
    total_weight = sum(weights.values())
    random_number = random.uniform(0, total_weight)
    cumulative_weight = 0
    for key, weight in weights.items():
        cumulative_weight += weight
        if random_number < cumulative_weight:
            return key

# Example usage:
sampled_year = weighted_random_choice(year_weights)
print(f"The sampled year is: {sampled_year}")

 random_number = random.uniform(0, total_weight)
1. random Module
This line uses the random module in Python, which provides various functions to generate random numbers and perform random operations.

2. uniform(a, b) Function
The uniform(a, b) function generates a random floating-point number 
x
x
such that 
a
≤
x
<
b
a≤x<b
.

In this case, it generates a number between 0 and total_weight.

3. 0 and total_weight
0: This is the lower bound of the range. The generated random number will be at least 0.

total_weight: This is the upper bound of the range. It represents the sum of all weights in the year_weights dictionary. The generated random number will be less than total_weight.

Purpose in Context
In the context of the function that samples a year based on weights:

Random Selection: The line generates a random number that will be used to determine which year to select based on its weight.

Cumulative Weighting: After generating this random number, the code iterates through the cumulative weights of each year. When the cumulative weight exceeds this random number, it identifies that year as the selected one.




'''

# Generate a random date based on custom probabilities
def generate_custom_date(year_weights):
    year = random.choices(list(year_weights.keys()), weights=list(year_weights.values()))[0]
    month = random.randint(1, 12)
    day = random.randint(1, 28)  # Assuming all months have 28 days for simplicity
    return fake.date_time_between(start_date=datetime(year, 1, 1), end_date=datetime(year, 12, 31))


'''
This code snippet is designed to generate a somewhat realistic, but still fake, datetime object. It does this by:

Sampling a year from a predefined set of years, using a weighting scheme to favor certain years.

Generating a random month (1-12).

Generating a random day (1-28), with the simplification that all months are treated as having 28 days.

Using a Faker library function to create a datetime object within the selected year, effectively overriding the random month/day with a faker-generated time.

Line-by-Line Explanation

year = random.choices(list(year_weights.keys()), weights=list(year_weights.values()))

year_weights.keys(): This retrieves the keys from the year_weights dictionary (which are the years, like 2015, 2016, etc.).

list(year_weights.keys()): This converts the keys (which are initially in a "view" object) into a Python list. This is necessary because random.choices expects 
a list as its input for the population to choose from.

year_weights.values(): This retrieves the values from the year_weights dictionary (which are the weights associated with each year).

list(year_weights.values()): This converts the values (the weights) into a list.

random.choices(..., weights=...): This is the core of the weighted random selection. The random.choices() function (note the plural "choices") selects elements 
from a population (the list of years) with a given set of weights. Because random.choices returns a list of selections (even if you're only asking for one), we need to extract the first element.

``: This extracts the first element from the list returned by random.choices. Since we're only asking for one selection, the list will only contain one element: 
the randomly chosen year. This year is then assigned to the variable year.

month = random.randint(1, 12)

random.randint(a, b): This function from the random module generates a random integer between a and b (inclusive).

1, 12: Specifies the range of possible months (1 for January, 12 for December). So, this line randomly selects a month between 1 and 12 and assigns it to the variable month.

day = random.randint(1, 28) # Assuming all months have 28 days for simplicity

This line is very similar to the previous line, but it generates a random day between 1 and 28.

# Assuming all months have 28 days for simplicity: This is a crucial comment! It acknowledges that this code is making a simplification. In reality, 
# months have different numbers of days (28, 29, 30, or 31). This simplification is made to keep the code simpler, but it does mean that the generated dates will not be perfectly realistic. 
# There will be no dates like January 29th, 30th, or 31st generated by this code, and no dates after February 28th in February.

return fake.date_time_between(start_date=datetime(year, 1, 1), end_date=datetime(year, 12, 31))

datetime(year, 1, 1): This creates a datetime object representing January 1st of the randomly selected year. It uses the datetime class from the datetime module.

datetime(year, 12, 31): This creates a datetime object representing December 31st of the randomly selected year.

fake.date_time_between(start_date=..., end_date=...): This uses a function from the Faker library (presumably you have a Faker instance named fake). 
This function generates a random datetime object between the specified start_date and end_date. This part overrides the random month and day generated earlier, 
creating a more realistic (though still fake) timestamp within the chosen year. The key point is that Faker is used to generate a random date within the selected year, 
instead of relying on the simplified month/day generation.
'''
def generate_salary(department, job_title):
    salary_dict = {
            'HR': {
                'HR Manager': np.random.randint(60000, 90000),
                'HR Coordinator': np.random.randint(50000, 60000),
                'Recruiter': np.random.randint(50000, 70000),
                'HR Assistant': np.random.randint(50000, 60000)
            },
            'IT': {
                'IT Manager': np.random.randint(80000, 120000),
                'Software Developer': np.random.randint(70000, 95000),
                'System Administrator': np.random.randint(60000, 90000),
                'IT Support Specialist': np.random.randint(50000, 60000)
            },
            'Sales': {
                'Sales Manager': np.random.randint(70000, 110000),
                'Sales Consultant': np.random.randint(60000, 90000),
                'Sales Specialist': np.random.randint(50000, 80000),
                'Sales Representative': np.random.randint(50000, 70000)
            },
            'Marketing': {
                'Marketing Manager': np.random.randint(70000, 100000),
                'SEO Specialist': np.random.randint(50000, 80000),
                'Content Creator': np.random.randint(50000, 60000),
                'Marketing Coordinator': np.random.randint(50000, 70000)
            },
            'Finance': {
                'Finance Manager': np.random.randint(80000, 120000),
                'Accountant': np.random.randint(50000, 80000),
                'Financial Analyst': np.random.randint(60000, 90000),
                'Accounts Payable Specialist': np.random.randint(50000, 60000)
            },
            'Operations': {
                'Operations Manager': np.random.randint(70000, 100000),
                'Operations Analyst': np.random.randint(50000, 80000),
                'Logistics Coordinator': np.random.randint(50000, 60000),
                'Inventory Specialist': np.random.randint(50000, 60000)
            },
            'Customer Service': {
                'Customer Service Manager': np.random.randint(60000, 90000),
                'Customer Service Representative': np.random.randint(50000, 60000),
                'Support Specialist': np.random.randint(50000, 60000),
                'Help Desk Technician': np.random.randint(50000, 80000)
            }
        }
    return salary_dict[department][job_title]

# Generate the dataset
data = []

for _ in range(num_records):
    employee_id = f"00-{random.randint(10000000, 99999999)}"
    first_name = fake.first_name()
    last_name = fake.last_name()
    gender = np.random.choice(['Female', 'Male'], p=[0.46, 0.54])
    state = np.random.choice(states, p=state_prob)
    city = np.random.choice(states_cities[state])
    hiredate = generate_custom_date(year_weights)
      #termdate
    department = np.random.choice(departments, p=departments_prob)
    job_title  = np.random.choice(jobtitles[department], p=jobtitles_prob[department])
    education_level = np.random.choice(education_mapping[job_title])
    performance_rating = np.random.choice(['Excellent', 'Good', 'Satisfactory', 'Needs Improvement'], p=[0.12, 0.5, 0.3, 0.08])
    overtime = np.random.choice(['Yes', 'No'], p=[0.3, 0.7])
    salary = generate_salary(department, job_title)

    data.append([
        employee_id,
        first_name,
        last_name,
        gender,
        state,
        city,
        hiredate,
        department,
        job_title,
        education_level,
        salary,
        performance_rating,
        overtime
    ])

## Create DataFrame
columns = [
     'employee_id',
     'first_name',
     'last_name',
     'gender',
     'state',
     'city',
     'hiredate',
     'department',
     'job_title',
     'education_level',
     'salary',
     'performance_rating',
     'overtime'
    ]


df = pd.DataFrame(data, columns=columns)

# Add Birthdate
def generate_birthdate(row):
    age_distribution = {
        'under_25': 0.11,
        '25_34': 0.25,
        '35_44': 0.31,
        '45_54': 0.24,
        'over_55': 0.09
    }
    age_groups = list(age_distribution.keys())
    age_probs = list(age_distribution.values())
    age_group = np.random.choice(age_groups, p=age_probs)

    if any('Manager' in title for title in row['job_title']):
        age = np.random.randint(30, 65)
    elif row['education_level'] == 'PhD':
        age = np.random.randint(27, 65)
    elif age_group == 'under_25':
         age = np.random.randint(20, 25)
    elif age_group == '25_34':
        age = np.random.randint(25, 35)
    elif age_group == '35_44':
        age = np.random.randint(35, 45)
    elif age_group == '45_54':
        age = np.random.randint(45, 55)
    else:
        age = np.random.randint(56, 65)

    birthdate = fake.date_of_birth(minimum_age=age, maximum_age=age)
    return birthdate

# Apply the function to generate birthdates
df['birthdate'] = df.apply(generate_birthdate, axis=1)

# Terminations
# Define termination distribution
year_weights = {
    2015: 5,
    2016: 7,
    2017: 10,
    2018: 12,
    2019: 9,
    2020: 10,
    2021: 20,
    2022: 10,
    2023: 7,
    2024: 10
}

# Calculate the total number of terminated employees
total_employees = num_records
termination_percentage = 0.112  # 11.2%
total_terminated = int(total_employees * termination_percentage)

# Generate termination dates based on distribution
termination_dates = []
for year, weight in year_weights.items():
    num_terminations = int(total_terminated * (weight / 100))
    termination_dates.extend([year] * num_terminations)

# Randomly shuffle the termination dates
random.shuffle(termination_dates)

# Assign termination dates to terminated employees
terminated_indices = df.index[:total_terminated]
for i, year in enumerate(termination_dates[:total_terminated]):
    df.at[terminated_indices[i], 'termdate'] = datetime(year, 1, 1) + timedelta(days=random.randint(0, 365))


'''
terminated_indices = df.index[:total_terminated]

df.index: This retrieves the index of the DataFrame df. The index typically represents the row labels of the DataFrame.

df.index[:total_terminated]: This slices the index to get the first total_terminated indices. This means you are selecting the first total_terminated rows from the DataFrame.

terminated_indices: This variable now holds a list of indices corresponding to the rows that will be updated with termination dates.

for i, year in enumerate(termination_dates[:total_terminated]):

termination_dates[:total_terminated]: This slices the termination_dates list (or array) to get only the first total_terminated years. This assumes that termination_dates contains a list of years when terminations occurred.

enumerate(...): This function iterates over the sliced list of years, providing both an index (i) and the corresponding year (year) for each iteration.

The loop will run for each year in the sliced termination_dates, allowing you to process each termination year individually.

df.at[terminated_indices[i], 'termdate'] = ...

df.at[...]: This is a way to access a specific cell in the DataFrame using label-based indexing. It allows you to set a value at a specific row and column.

terminated_indices[i]: This accesses the index of the row that corresponds to the current iteration's index (i). It retrieves which row in df should be updated with a termination date.

'termdate': This specifies the column in which you want to assign the termination date. The code assumes that this column exists in your DataFrame.

datetime(year, 1, 1) + timedelta(days=random.randint(0, 365))

datetime(year, 1, 1): This creates a datetime object representing January 1st of the specified year.

random.randint(0, 365): This generates a random integer between 0 and 365, representing a random number of days within that year.

timedelta(days=...): The timedelta class from Python's datetime module represents a duration or difference between two dates.
 Here, it is used to create a duration based on the random number of days generated.

The entire expression calculates a random date within that year by adding a random number of days (from 0 to 365) to January 1st.
'''

# Assign None to termdate for employees who are not terminated
df['termdate'] = df['termdate'].where(df['termdate'].notnull(), None)

# Ensure termdate is at least 6 months after hiredat
df['termdate'] = df.apply(lambda row: row['hiredate'] + timedelta(days=180) if row['termdate'] and row['termdate'] < row['hiredate'] + timedelta(days=180) else row['termdate'], axis=1)

'''
This code is designed to update the 'termdate' column in a pandas DataFrame (df). It sets the termination date to be 180 days after the hire date if the current 
termination date is either not set or is earlier than that calculated date.

Breakdown of Each Component
df.apply(...):

The apply() function in pandas is used to apply a function along a particular axis of the DataFrame (rows or columns). In this case, axis=1 indicates that the function will be applied to each row.

lambda row: ...:

This defines an anonymous (lambda) function that takes a single argument, row, which represents each row of the DataFrame as the function iterates through it.

row['hiredate'] + timedelta(days=180):

This part calculates a date that is 180 days after the hire date of the employee.

row['hiredate']: Accesses the hire date for the current row.

timedelta(days=180): Creates a timedelta object representing a duration of 180 days.

The expression effectively computes the date that is 180 days after the hire date.

if row['termdate'] and row['termdate'] < row['hiredate'] + timedelta(days=180):

This conditional statement checks two things:

row['termdate']: Checks if there is a value in the 'termdate' column (i.e., it is not None or NaN). If there is no termination date, this condition will evaluate to False.

row['termdate'] < row['hiredate'] + timedelta(days=180): Checks if the existing termination date is earlier than 180 days after the hire date.

If both conditions are True, it means that there is a termination date set, and it occurs before the calculated date (180 days after hire).

else row['termdate']:

If either condition in the if statement is not met (i.e., if there is no termination date or if it’s not less than 180 days after hire), 
then this part of the expression returns the existing termination date without any modification.

df['termdate'] = ...:

This assigns the result of the apply() function back to the 'termdate' column of the DataFrame. The entire column will be updated based on the logic defined in the lambda function.
'''

education_multiplier = {
    'High School': {'Male': 1.03, 'Female': 1.0},
    "Bachelor": {'Male': 1.115, 'Female': 1.0},
    "Master": {'Male': 1.0, 'Female': 1.07},
    'PhD': {'Male': 1.0, 'Female': 1.17}
}

'''
Salary Multipliers:

The multipliers indicate how much an individual's salary may be adjusted based on their education level and gender.

For example:

A male with a high school education has a multiplier of 1.03, meaning their salary is increased by 3% compared to a baseline.

A female with a master's degree has a multiplier of 1.07, meaning her salary is increased by 7% compared to a baseline.


'''


# Function to calculate age from birthdate
def calculate_age(birthdate):
    today = pd.Timestamp('today')
    age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    return age

'''
today.year - birthdate.year:

today.year: This gets the current year from a datetime or date object named today.

birthdate.year: This gets the year of birth from a datetime or date object named birthdate.

The subtraction calculates the difference in years between the current year and the birth year. This gives a preliminary age.

((today.month, today.day) < (birthdate.month, birthdate.day)):

today.month: Gets the current month.

today.day: Gets the current day of the month.

birthdate.month: Gets the birth month.

birthdate.day: Gets the birth day.

(today.month, today.day) < (birthdate.month, birthdate.day): This is a tuple comparison. Python compares tuples element by element. It first compares the months.
 If the current month is earlier than the birth month, then the whole expression is True. If the current month is later than the birth month, then the whole expression is False.
 If the months are the same, then it compares the days. If the current day is earlier than the birth day, the expression is True. Otherwise, it's False. 
 In essence, this checks if the person's birthday has already occurred this year.

The result of this comparison will be either True or False. In Python, True is equivalent to 1 and False is equivalent to 0 in numerical contexts.

age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day)):

This line combines the two parts. It subtracts the result of the tuple comparison (either 0 or 1) from the initial year difference.

If the birthday has already occurred this year, the tuple comparison will be False (0), so the age will simply be the difference in years.

If the birthday has not yet occurred this year, the tuple comparison will be True (1), so the age will be one less than the difference in years.
'''

# Function to calculate the adjusted salary
def calculate_adjusted_salary(row):
    base_salary = row['salary']
    gender = row['gender']
    education = row['education_level']
    age = calculate_age(row['birthdate'])

    # Apply education multiplier
    multiplier = education_multiplier.get(education, {}).get(gender, 1.0)
    adjusted_salary = base_salary * multiplier

    '''
    multiplier = education_multiplier.get(education, {}).get(gender, 1.0):

This line attempts to retrieve the salary multiplier based on the education and gender variables.

education_multiplier.get(education, {}): This retrieves the dictionary corresponding to the specified education.
 If the education key does not exist, it returns an empty dictionary ({}) to avoid errors.

.get(gender, 1.0): This further retrieves the multiplier for the specified gender. If the gender key does not exist in the inner dictionary, 
it defaults to 1.0, meaning no adjustment.

adjusted_salary = base_salary * multiplier:

This line calculates the adjusted salary by multiplying the base salary (base_salary) by the retrieved multiplier.
    '''

    # Apply age increment (between 0.1% and 0.3% per year of age)
    age_increment = 1 + np.random.uniform(0.001, 0.003) * age
    adjusted_salary *= age_increment

    # Ensure the adjusted salary is not lower than the base salary
    adjusted_salary = max(adjusted_salary, base_salary)

    # Round the adjusted salary to the nearest integer
    return round(adjusted_salary)

# Apply the function to the DataFrame
df['salary'] = df.apply(calculate_adjusted_salary, axis=1)

#The apply() function in pandas is used to apply a function along a particular axis of the DataFrame.

#The axis=1 argument specifies that the function should be applied to each row (as opposed to each column, which would be specified with axis=0).
#This part assigns the result of the apply() operation back to a new or existing column named 'salary' in the DataFrame.
#Each entry in the 'salary' column will be populated with the adjusted salary value calculated by the calculate_adjusted_salary function for each corresponding row.



# Convert 'hiredate' and 'birthdate' to datetime
df['hiredate'] = pd.to_datetime(df['hiredate']).dt.date
df['birthdate'] = pd.to_datetime(df['birthdate']).dt.date
df['termdate'] = pd.to_datetime(df['termdate']).dt.date

#used to convert specific columns in a pandas DataFrame (df) to date objects
#After converting to datetime, the .dt accessor allows you to access various properties of datetime objects.
#.date: This converts the datetime objects into Python date objects, which only store the date (year, month, day) 
# without any time information (hours, minutes, seconds). This is useful if you only need the date and want to discard any time component.

print(df)

# Save to CSV
df.to_csv('HumanResources.csv', index=False)