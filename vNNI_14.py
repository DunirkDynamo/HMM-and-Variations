import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import *
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import math


class State:
    def __init__(self, mean, variance, duration_dist=None, duration_dist_params=None):
        # For the normally distributed current samples:
        self.mean                 = mean
        self.variance             = variance 
        # For the distribution of step durations:
        self.duration_dist        = duration_dist
        self.duration_dist_params = duration_dist_params

                        
    def prob_n(self, n, sample_rate):
        from scipy.integrate import quad
        t1 = quad(self.duration_dist, n/sample_rate, (n+1)/sample_rate, tuple(self.duration_dist_params))[0]
        t2 = quad(self.duration_dist, (n-1)/sample_rate, n/sample_rate, tuple(self.duration_dist_params))[0]
        J  = (n+1) * t1 - (n-1) * t2

        def expectation_duration_dist(x):
            out = x*self.duration_dist(x, *self.duration_dist_params)
            return out
            

        t3 = quad(expectation_duration_dist, (n-1)/sample_rate, n/sample_rate)[0]
        t4 = quad(expectation_duration_dist, n/sample_rate    , (n+1)/sample_rate)[0]
        W  = (t3 - t4)*sample_rate

        output = J + W
        return output
    
    def joint_n_xvec(self, x, gamma, tol):

        # Computing the Normal portion of the joint probability vector (n, x), where x is itself a vector of length n
        n = len(x)
        delta = x - self.mean
        vhat  = np.dot(delta, delta)/n
        A     = 1./(2*pi*self.variance)**(n/2.)

        xdensity = A*np.exp(-0.5*n*vhat/self.variance)
        
        
        # Computing the probability of n
        ndensity = self.prob_n(n, 1./gamma)
        
        
        # Computing joint of n and x:
        joint_density = ndensity*xdensity
        
        return joint_density
    
    def log_joint_n_xvec(self, x, gamma, tol):
        from math import pi as pi
        # Computing the Normal portion of the joint probability vector (n, x), where x is itself a vector of length n
        n     = len(x)
        delta = x - self.mean
        vhat  = np.dot(delta, delta)/n
        #A     = 1./(2*pi*self.variance)**(n/2.)
        log_A = -n*np.log(2*pi*self.variance)/2
        
        #xdensity = A*np.exp(-0.5*n*vhat/self.variance)
        log_xdensity = log_A - n*vhat/2/self.variance
        
        # Computing the probability of n
        ndensity = self.prob_n(n, 1./gamma)
        # Computing joint of n and x:
        log_joint_density = np.log(ndensity) + log_xdensity
        return log_joint_density

    def draw(self, num_samples=1):
        # Generates a normally distributed random variable with the specified mean and variance
        return np.random.normal(loc=self.mean, scale = np.sqrt(self.variance), size = num_samples)

    def duration_instance(self):
        # Draws from a normal distribution based on the duration and duration_variance
        return self.rejection_sampling(self.duration_dist, bounds=(0.000001, 10), num_samples=1, args=self.duration_dist_params)[0]
    
    def rejection_sampling(density_func, bounds, num_samples=1, args=[]):
        '''Draws samples from a user-defined density function using rejection sampling.
        
        Parameters:
            density_func : function
                A function that takes a sample point and returns the density at that point.
            bounds : tuple
                A tuple (lower_bound, upper_bound) defining the range from which to sample.
            num_samples : int
                The number of samples to draw (default is 1).
        
        Returns:
            np.array
                An array of samples drawn from the distribution defined by the density function.
        '''
        # Determine the maximum value of the density function in the given bounds
        lower_bound, upper_bound = bounds
        sample_points = np.linspace(lower_bound, upper_bound, 1000)
        
        if args:
            max_density = max(density_func(x, *args) for x in sample_points)
        else:
            max_density = max(density_func(x) for x in sample_points)
        # List to store the accepted samples
        samples = []
        
        while len(samples) < num_samples:
            # Sample a candidate from the uniform distribution in the given range
            candidate = np.random.uniform(lower_bound, upper_bound)
            
            # Sample a uniform random value between 0 and max_density
            u = np.random.uniform(0, max_density)
            
            # Evaluate the density at the candidate point
            if args:
                density = density_func(candidate, *args)
            else:
                density = density_func(candidate)
            
            # Accept the candidate with probability proportional to its density
            if u < density:
                samples.append(candidate)
        
        return np.array(samples)



    def emission_probability(self, obs):
        """Calculates the Gaussian emission probability for a given observation."""
        return (1 / ( np.sqrt(self.variance *2 * np.pi))) * np.exp(-0.5 * ((obs - self.mean) / np.sqrt(self.variance)) ** 2)


class DataFramePlotterApp:
    def __init__(self, root):
        """
        Purpose:
            Initialize the GUI application, setting up all frames, widgets, and their layouts.
        
        Inputs:
            root (tk.Tk): The root Tkinter window where the application runs.

        Outputs:
            None. This constructor sets up the GUI's initial state.
        """
        self.root = root
        self.root.title("DataFrame Plotter")



        # DataFrame loading frame
        self.load_frame = ttk.LabelFrame(root, text="Load DataFrame")
        self.load_frame.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.load_button = ttk.Button(self.load_frame, text="Load DataFrame", command=self.load_dataframe)
        self.load_button.grid(row=0, column=0, padx=5, pady=5)

        # Columns selection frame
        self.select_frame = ttk.LabelFrame(root, text="Select Columns")
        self.select_frame.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        self.column1_label = ttk.Label(self.select_frame, text="Column 1:")
        self.column1_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.column1_combobox = ttk.Combobox(self.select_frame, state="readonly")
        self.column1_combobox.grid(row=0, column=1, padx=5, pady=5)

        self.column2_label = ttk.Label(self.select_frame, text="Column 2:")
        self.column2_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.column2_combobox = ttk.Combobox(self.select_frame, state="readonly")
        self.column2_combobox.grid(row=1, column=1, padx=5, pady=5)

        # Plot button
        self.plot_button = ttk.Button(self.select_frame, text="Plot", command=self.plot_data)
        self.plot_button.grid(row=2, column=0, padx=10, pady=10)

        # DataFrame display frame (initially hidden)
        self.display_frame = ttk.LabelFrame(root, text="DataFrame Preview")
        self.display_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
        self.display_frame.grid_remove()

        # DataFrame preview label
        self.preview_label = ttk.Label(self.display_frame, text="First 10 rows of DataFrame:")
        self.preview_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # DataFrame preview text
        self.preview_text = tk.Text(self.display_frame, height=10, width=100)
        self.preview_text.grid(row=1, column=0, padx=5, pady=5)

        # Plot Frame
        self.plot_frame = ttk.LabelFrame(root, text="Plot Frame")
        self.plot_frame.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")

        # Data processing frame (initially hidden)
        self.processing_frame = ttk.LabelFrame(root, text="Data Processing")
        self.processing_frame.grid(row=5, column=0, padx=10, pady=10, sticky="nsew")
        self.processing_frame.grid_remove()


        # New Model Frame

        self.model_frame = ttk.LabelFrame(root, text="Model Frame")
        self.model_frame.grid(row=3, column=1, rowspan=2, columnspan = 10, padx=10, pady=10, sticky="nsew")
        
        # Add Define HMM/HSMM button
        self.define_button = ttk.Button(
            self.model_frame, text="Define HMM/HSMM", command=self.show_hmm_hsmm_setup
        )
        self.define_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Initialize variables
        self.number_of_states = None
        self.uniform_var        = tk.BooleanVar(value=False)
        self.state_means        = []
        self.state_vars         = []
        self.sample_rate        = None # this is a variable that must be defined for some of the decoder algorithms
        self.sample_rate        = None
        self.max_chunk_size     = None
        self.changepoints       = [] # this is used by the 2-stage vector-state HMM



        # Decode Button (i.e. hidden state estimation button)
        self.fitModelChoice = ''



        def get_selected(choice):
            """
            Purpose:
                Update the selected model type and toggle the visibility of the "Select" button.

            Inputs:
                choice (str): The selected model type.

            Outputs:
                None. Modifies the state of the application based on the selection.
            """
            self.fitModelChoice = self.clicked.get()
            if self.fitModelChoice != "Fit Model":
                self.select_button.grid(row=1, padx=5, pady=5)
            else:
                self.select_button.grid_remove()



        self.modelOptions = [
            'Viterbi --> HMM',
            'Viterbi --> Time-Dependent Pseudo-HMM (Strong)',
            'Viterbi --> Time-Dependent Pseudo-HMM (Weak)',
            'Viterbi --> Averaged Chunked HMM',
            'Viterbi --> HSMM',
            'Forward-Backward --> HMM',
            'Viterbi ---> 2-Stage Changepoint-Vector-State HMM'
        ]


        self.clicked = StringVar()
        self.clicked.set("Fit Model")
        self.fitModelButton = OptionMenu(self.processing_frame, self.clicked, *self.modelOptions, command=get_selected)
        self.fitModelButton.grid(row=0, padx=5, pady=5)

        self.clicked = StringVar()
        self.clicked.set("Fit Model")
        self.fitModelButton = OptionMenu(self.processing_frame, self.clicked, *self.modelOptions, command=get_selected)
        self.fitModelButton.grid(row=0, padx=5, pady=5)


        # Select model button (initially hidden)
        self.select_button = ttk.Button(self.processing_frame, text="Select", command=self.select_model)
        self.select_button.grid_remove()

        # Execute model button (initially hidden)
        #self.execute_button = ttk.Button(self.processing_frame, text="Execute Model", command=self.execute_model)
        #self.execute_button.grid_remove()

        # DataFrame
        self.df = None # gets populated after user imports data using load_dataframe which is accessed by clicking "Load Dataframe" button

    def clear_entries(self):
        """
        Purpose:
            Clear all entry fields and reset values to default.

        Outputs:
            None. Resets instance variables and hides input fields.
        """
        self.sample_rate    = None
        self.max_chunk_size = None
        self.changepoints   = []

        for widget in self.processing_frame.winfo_children():
            if isinstance(widget, (ttk.Entry, ttk.Label, ttk.Button)) and widget not in [self.select_button, self.export_button]:
                widget.grid_remove()

    def load_dataframe(self):
        """
        Purpose:
            Allow the user to load a DataFrame from a CSV or Excel file.

        Inputs:
            None. File is selected using a file dialog.

        Outputs:
            None. Updates the application's DataFrame state and GUI elements.
        """
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls")])

        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.populate_columns_comboboxes()
                messagebox.showinfo("Success", "DataFrame loaded successfully!")
                self.show_dataframe_preview()
            except Exception as e:
                messagebox.showerror("Error", f"Error loading DataFrame: {e}")

    def populate_columns_comboboxes(self):
        """
        Purpose:
            Populate the column selection comboboxes with column names from the loaded DataFrame.

        Inputs:
            None. Operates on the `self.df` DataFrame.

        Outputs:
            None. Updates the column comboboxes in the GUI.
        """
        columns = self.df.columns.tolist()
        self.column1_combobox["values"] = columns
        self.column2_combobox["values"] = columns

    def show_dataframe_preview(self):
        """
        Purpose:
            Display a preview of the first 10 rows of the loaded DataFrame in a text widget.

        Inputs:
            None. Operates on the `self.df` DataFrame.

        Outputs:
            None. Updates the preview text widget in the GUI.
        """
        self.preview_text.delete(1.0, tk.END)
        preview_text = self.df.head(10).to_string(index=False)
        self.preview_text.insert(tk.END, preview_text)
        self.display_frame.grid()
        self.processing_frame.grid()

    def plot_data(self):
        """
        Purpose:
            Generate and display a scatter plot based on two selected columns from the DataFrame.

        Inputs:
            None. Uses selected columns from comboboxes.

        Outputs:
            None. Updates the plot frame in the GUI with the generated plot.
        """
        if self.df is None:
            messagebox.showerror("Error", "Please load a DataFrame first!")
            return

        column1 = self.column1_combobox.get()
        column2 = self.column2_combobox.get()

        if not column1 or not column2:
            messagebox.showerror("Error", "Please select both columns!")
            return

        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            plt.scatter(self.df[column1], self.df[column2])
            plt.xlabel(column1)
            plt.ylabel(column2)
            plt.title(f"Scatter Plot of {column1} vs {column2}")
            plt.grid(True)
            self.plot_to_frame(fig)
        except Exception as e:
            messagebox.showerror("Error", f"Error plotting data: {e}")

    def plot_to_frame(self, fig):
        """
        Purpose:
            Embed a matplotlib plot into the plot frame in the GUI.

        Inputs:
            fig (matplotlib.figure.Figure): The figure to embed.

        Outputs:
            None. Embeds the plot into the GUI.
        """
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


    def show_hmm_hsmm_setup(self):
        """Handle the setup for defining HMM/HSMM."""
        self.define_button.grid_remove()

        # Add entry for number of states
        self.num_states_label = ttk.Label(self.model_frame, text="Number of States:")
        self.num_states_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.num_states_entry = ttk.Entry(self.model_frame)
        self.num_states_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        self.num_states_confirm_button = ttk.Button(
            self.model_frame, text="Confirm", command=self.confirm_num_states
        )
        self.num_states_confirm_button.grid(row=1, column=2, padx=5, pady=5, sticky="w")

    def confirm_num_states(self):
        """Validate and store the number of states."""
        try:
            num_states = int(self.num_states_entry.get())
            if num_states <= 0:
                raise ValueError("Number of states must be a positive integer.")

            self.number_of_states = num_states
            self.num_states_label.grid_remove()
            self.num_states_entry.grid_remove()
            self.num_states_confirm_button.grid_remove()

            
            self.create_initial_probabilities_section() # Add section for definining initial probabilities
            self.create_emission_parameters_section()   # add section for defining emission distributions
            self.create_transition_matrix_section()     # add section for defining transition matrix
            self.create_hsmm_parameters_section()       # Add subframe for HSMM duration distribution parameters 
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def confirm_initial_probabilities(self):
        """Validate and display the initial probabilities."""
        try:
            raw_probs = self.init_prob_entry.get().split(",")
            if self.uniform_var.get():
                if len(raw_probs) != 1:
                    raise ValueError("Enter exactly one probability for uniform distribution.")
                prob = float(raw_probs[0])
                if prob <= 0 or prob > 1:
                    raise ValueError("Probability must be in the range (0, 1].")
                self.initial_probabilities = [prob] * self.number_of_states
            else:
                probs = [float(p) for p in raw_probs]
                if len(probs) != self.number_of_states:
                    raise ValueError(
                        f"Number of probabilities must match number of states ({self.number_of_states})."
                    )
                if not math.isclose(sum(probs), 1.0, rel_tol=1e-6):
                    raise ValueError("Probabilities must sum to 1.")
                self.initial_probabilities = probs

            # Remove existing widgets in the frame and display probabilities
            for widget in self.initial_prob_frame.winfo_children():
                widget.destroy()

            self.init_prob_display_label = ttk.Label(self.initial_prob_frame, text="Initial Probabilities:")
            self.init_prob_display_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

            scroll_canvas = tk.Canvas(self.initial_prob_frame, height=100)
            scroll_canvas.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

            scrollbar = ttk.Scrollbar(self.initial_prob_frame, orient="vertical", command=scroll_canvas.yview)
            scrollbar.grid(row=1, column=2, sticky="ns")

            scrollable_frame = ttk.Frame(scroll_canvas)
            scroll_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            scroll_canvas.configure(yscrollcommand=scrollbar.set)

            def update_scroll_region(event):
                scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

            scrollable_frame.bind("<Configure>", update_scroll_region)

            for i, prob in enumerate(self.initial_probabilities):
                prob_label = ttk.Label(
                    scrollable_frame, text=f"State-{i}: {prob:.4f}"
                )
                prob_label.pack(anchor="w")

            messagebox.showinfo("Success", "Initial probabilities set successfully!")
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def confirm_state_means(self):
        """Validate and display the state means with a header."""
        try:
            raw_means = self.state_means_entry.get().split(",")
            means = [float(m) for m in raw_means]
            if len(means) != self.number_of_states:
                raise ValueError(
                    f"Number of means must match number of states ({self.number_of_states})."
                )
            self.state_means = means

            # Clear existing means widgets
            for widget in self.shared_scrollable_frame.winfo_children():
                widget.destroy()

            # Add header for State Means
            header_label = ttk.Label(
                self.shared_scrollable_frame, text="State Means", font=("Arial", 10, "bold")
            )
            header_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

            # Add updated means to the shared scrollable frame
            for i, mean in enumerate(self.state_means):
                mean_label = ttk.Label(
                    self.shared_scrollable_frame, text=f"State-{i}: {mean:.4f}", name=f"means-{i}"
                )
                mean_label.grid(row=i + 1, column=0, padx=5, pady=5, sticky="w")

            messagebox.showinfo("Success", "State means set successfully!")
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def confirm_state_vars(self):
        """Validate and display the state variances with a header."""
        try:
            raw_vars = self.state_vars_entry.get().split(",")
            vars     = [float(s) for s in raw_vars]
            if len(vars) != self.number_of_states:
                raise ValueError(
                    f"Number of variances must match number of states ({self.number_of_states})."
                )
            self.state_vars = vars

            # Add header for State variances
            header_label = ttk.Label(
                self.shared_scrollable_frame, text="State Variances", font=("Arial", 10, "bold")
            )
            header_label.grid(row=0, column=1, padx=20, pady=5, sticky="w")

            # Add updated varainces to the shared scrollable frame
            for i, var in enumerate(self.state_vars):
                var_label = ttk.Label(
                    self.shared_scrollable_frame, text=f"State-{i}: {var:.4f}", name=f"vars-{i}"
                )
                var_label.grid(row=i + 1, column=1, padx=20, pady=5, sticky="w")

            messagebox.showinfo("Success", "State variances set successfully!")
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def display_parameters(self, parameters, title):
        """Display parameters with a scrollable frame."""
        display_label = ttk.Label(self.model_frame, text=title)
        display_label.grid(row=6, column=0, padx=5, pady=5, sticky="w")

        if len(parameters) > 5:
            scroll_canvas = tk.Canvas(self.model_frame, height=100)
            scroll_canvas.grid(row=7, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

            scrollbar = ttk.Scrollbar(self.model_frame, orient="vertical", command=scroll_canvas.yview)
            scrollbar.grid(row=7, column=2, sticky="ns")

            scrollable_frame = ttk.Frame(scroll_canvas)

            scroll_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            scroll_canvas.configure(yscrollcommand=scrollbar.set)

            def update_scroll_region(event):
                scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

            scrollable_frame.bind("<Configure>", update_scroll_region)

            for i, param in enumerate(parameters):
                param_label = ttk.Label(
                    scrollable_frame,
                    text=f"{title} of State-{i}: {param:.4f}"
                )
                param_label.pack(anchor="w")
        else:
            for i, param in enumerate(parameters):
                param_label = ttk.Label(
                    self.model_frame,
                    text=f"{title} of State-{i}: {param:.4f}"
                )
                param_label.grid(row=7 + i, column=0, padx=5, pady=2, sticky="w")



    def create_hsmm_parameters_section(self):
        """Create a subframe for entering HSMM parameters."""
        # Create a LabelFrame for HSMM Parameters
        self.hsmm_frame = ttk.LabelFrame(self.model_frame, text="HSMM Parameters")
        self.hsmm_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

        # Parameter - d
        self.hsmm_d_label = ttk.Label(self.hsmm_frame, text="State Parameters - d\n(comma-separated):")
        self.hsmm_d_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.hsmm_d_entry = ttk.Entry(self.hsmm_frame, width=50)
        self.hsmm_d_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.hsmm_d_button = ttk.Button(
            self.hsmm_frame, text="Confirm", command=lambda: self.confirm_hsmm_parameter(self.hsmm_d_entry, "d")
        )
        self.hsmm_d_button.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Parameter - v
        self.hsmm_v_label = ttk.Label(self.hsmm_frame, text="State Parameters - v\n(comma-separated):")
        self.hsmm_v_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.hsmm_v_entry = ttk.Entry(self.hsmm_frame, width=50)
        self.hsmm_v_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        self.hsmm_v_button = ttk.Button(
            self.hsmm_frame, text="Confirm", command=lambda: self.confirm_hsmm_parameter(self.hsmm_v_entry, "v")
        )
        self.hsmm_v_button.grid(row=1, column=2, padx=5, pady=5, sticky="w")

        # Parameter - s
        self.hsmm_s_label = ttk.Label(self.hsmm_frame, text="State Parameters - s\n(comma-separated):")
        self.hsmm_s_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        self.hsmm_s_entry = ttk.Entry(self.hsmm_frame, width=50)
        self.hsmm_s_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        self.hsmm_s_button = ttk.Button(
            self.hsmm_frame, text="Confirm", command=lambda: self.confirm_hsmm_parameter(self.hsmm_s_entry, "s")
        )
        self.hsmm_s_button.grid(row=2, column=2, padx=5, pady=5, sticky="w")

        self.hsmm_parameters = {"d": [], "v": [], "s": []}

    def confirm_hsmm_parameter(self, entry, param_type):
        """Validate and store HSMM parameters."""
        try:
            raw_values = entry.get().split(",")
            values = [float(value.strip()) for value in raw_values]

            if len(values) != self.number_of_states:
                raise ValueError(f"You must enter exactly {self.number_of_states} values.")

            if any(value <= 0 for value in values):
                raise ValueError("All values must be strictly positive and non-zero.")

            self.hsmm_parameters[param_type] = values
            messagebox.showinfo("Success", f"Parameter {param_type} set successfully!")
        except ValueError as e:
            messagebox.showerror("Error", str(e))


    def create_initial_probabilities_section(self):
        """Create a subframe for entering initial probabilities."""
        # Create subframe for initial probabilities
        self.initial_prob_frame = ttk.LabelFrame(self.model_frame, text="Initial Probabilities")
        self.initial_prob_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

        self.init_prob_label = ttk.Label(
            self.initial_prob_frame, text="Initial Probabilities of States (comma-separated):"
        )
        self.init_prob_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.init_prob_entry = ttk.Entry(self.initial_prob_frame)
        self.init_prob_entry.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.uniform_radio = ttk.Checkbutton(
            self.initial_prob_frame,
            text="Uniform Initial Probabilities",
            variable=self.uniform_var,
            command=self.toggle_uniform_probabilities,
        )
        self.uniform_radio.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        self.init_prob_confirm_button = ttk.Button(
            self.initial_prob_frame, text="Confirm", command=self.confirm_initial_probabilities
        )
        self.init_prob_confirm_button.grid(row=3, column=0, padx=5, pady=5, sticky="w")

    def create_emission_parameters_section(self):
        """Create a subframe for entering emission distribution parameters."""
        # Create subframe for emission parameters
        self.emission_param_frame = ttk.LabelFrame(self.model_frame, text="Emission Distribution Parameters")
        self.emission_param_frame.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

        # Means section
        self.state_means_label = ttk.Label(
            self.emission_param_frame, text="State Means (comma-separated):"
        )
        self.state_means_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.state_means_entry = ttk.Entry(self.emission_param_frame)
        self.state_means_entry.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.state_means_confirm_button = ttk.Button(
            self.emission_param_frame, text="Confirm", command=self.confirm_state_means
        )
        self.state_means_confirm_button.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        # Variances section
        self.state_vars_label = ttk.Label(
            self.emission_param_frame, text="State Variances (comma-separated):"
        )
        self.state_vars_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.state_vars_entry = ttk.Entry(self.emission_param_frame)
        self.state_vars_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        self.state_vars_confirm_button = ttk.Button(
            self.emission_param_frame, text="Confirm", command=self.confirm_state_vars
        )
        self.state_vars_confirm_button.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # Add a shared scrolling canvas to display the means and variances of the emission distributions
        self.shared_scroll_canvas = tk.Canvas(self.emission_param_frame, height=150)
        self.shared_scroll_canvas.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")  # Updated alignment

        self.shared_scrollbar = ttk.Scrollbar(
            self.emission_param_frame, orient="vertical", command=self.shared_scroll_canvas.yview
        )
        self.shared_scrollbar.grid(row=3, column=2, sticky="ns")  # Updated alignment

        self.shared_scrollable_frame = ttk.Frame(self.shared_scroll_canvas)
        self.shared_scroll_canvas.create_window((0, 0), window=self.shared_scrollable_frame, anchor="nw")
        self.shared_scroll_canvas.configure(yscrollcommand=self.shared_scrollbar.set)


    

    def create_transition_matrix_section(self):
        """Create a subframe for entering and confirming the transition matrix."""
        # Create a LabelFrame for the Transition Matrix
        self.transition_matrix_frame = ttk.LabelFrame(self.model_frame, text="Transition Matrix")
        self.transition_matrix_frame.grid(row=1, column=3, rowspan=2, padx=10, pady=10, sticky="nsew")

        # Configure grid layout for alignment
        self.transition_matrix_frame.columnconfigure(0, weight=1)
        self.transition_matrix_frame.columnconfigure(1, weight=2)
        # Add button to load matrix from file
        self.load_file_button = ttk.Button(
            self.transition_matrix_frame,
            text="Load Matrix from File",
            command=self.load_transition_matrix_from_file
        )
        self.load_file_button.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # Determine if scrolling is needed
        if self.number_of_states > 8:
            # Add scrollable canvas
            self.transition_canvas = tk.Canvas(self.transition_matrix_frame, height=200)
            self.transition_canvas.grid(row=0, column=0, columnspan=2, sticky="nsew")

            self.transition_scrollbar_v = ttk.Scrollbar(
                self.transition_matrix_frame, orient="vertical", command=self.transition_canvas.yview
            )
            self.transition_scrollbar_v.grid(row=0, column=2, sticky="ns")

            self.transition_scrollable_frame = ttk.Frame(self.transition_canvas)
            self.transition_canvas.create_window((0, 0), window=self.transition_scrollable_frame, anchor="nw")
            self.transition_canvas.configure(yscrollcommand=self.transition_scrollbar_v.set)

            self.transition_scrollable_frame.columnconfigure(0, weight=1)
            self.transition_scrollable_frame.columnconfigure(1, weight=2)

            parent_frame = self.transition_scrollable_frame
        else:
            # No scrolling needed; use the frame directly
            parent_frame = self.transition_matrix_frame

        self.transition_entries = []
        self.transition_confirm_buttons = []
        self.transition_matrix = [None] * self.number_of_states

        # Create entry boxes, labels, and confirm buttons
        for i in range(self.number_of_states):
            label = ttk.Label(
                parent_frame, text=f"State-{i} Transition Probabilities:\n(comma separated)"
            )
            label.grid(row=2 * i + 1, column=0, padx=5, pady=5, sticky="w")

            entry = ttk.Entry(parent_frame)
            entry.grid(row=2 * i + 1, column=1, padx=5, pady=5, sticky="ew")
            self.transition_entries.append(entry)

            button = ttk.Button(
                parent_frame,
                text="Confirm",
                command=lambda idx=i: self.confirm_transition_row(idx)
            )
            button.grid(row=2 * i + 1, column=2, padx=5, pady=5, sticky="ew")
            self.transition_confirm_buttons.append(button)

        if self.number_of_states > 5:
            # Adjust canvas scroll region
            parent_frame.update_idletasks()
            self.transition_canvas.configure(scrollregion=self.transition_canvas.bbox("all"))

    def confirm_transition_row(self, row_idx):
        """Validate and store the transition probabilities for a specific row."""
        try:
            # Parse the input and validate
            raw_probs = self.transition_entries[row_idx].get().split(",")
            probs = [float(p) for p in raw_probs]

            if len(probs) != self.number_of_states:
                raise ValueError(
                    f"Number of probabilities must match the number of states ({self.number_of_states})."
                )
            if any(p < 0 for p in probs):
                raise ValueError("Probabilities must be non-negative.")
            if not math.isclose(sum(probs), 1.0, rel_tol=1e-6):
                raise ValueError("Probabilities must sum to 1.")

            # Store the probabilities and disable the entry box and button
            self.transition_matrix[row_idx] = probs
            self.transition_entries[row_idx].config(state="disabled")
            self.transition_confirm_buttons[row_idx].config(state="disabled")

            # Check if all rows are confirmed
            if all(row is not None for row in self.transition_matrix):
                self.display_transition_matrix()

            messagebox.showinfo("Success", f"State-{row_idx} transition probabilities confirmed!")
        except ValueError as e:
            messagebox.showerror("Error", str(e))
    
    
    def load_transition_matrix_from_file(self):
        """Load the transition matrix from a CSV or Excel file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
        )
        if not file_path:
            return

        try:
            # Read the file based on extension
            if file_path.endswith(".csv"):
                import pandas as pd
                df = pd.read_csv(file_path, header=None)
            elif file_path.endswith(".xlsx"):
                import pandas as pd
                df = pd.read_excel(file_path, header=None)
            else:
                raise ValueError("Unsupported file format.")

            matrix = df.values.tolist()

            # Validate the matrix size
            if len(matrix) != self.number_of_states or any(len(row) != self.number_of_states for row in matrix):
                raise ValueError(
                    f"Matrix size must be {self.number_of_states}x{self.number_of_states}."
                )

            # Validate each row
            for row in matrix:
                if any(p < 0 for p in row):
                    raise ValueError("Probabilities must be non-negative.")
                if not math.isclose(sum(row), 1.0, rel_tol=1e-6):
                    raise ValueError("Each row must sum to 1.")

            # Store the matrix and update the display
            self.transition_matrix = matrix
            self.display_transition_matrix()

            messagebox.showinfo("Success", "Transition matrix loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    
    def display_transition_matrix(self):
        """Display the confirmed transition matrix as a grid."""
        # Clear existing widgets in the transition_matrix_frame
        for widget in self.transition_matrix_frame.winfo_children():
            widget.destroy()

        # Add a scrollable grid for the transition matrix
        canvas = tk.Canvas(self.transition_matrix_frame)
        canvas.grid(row=0, column=0, columnspan=2, sticky="nsew")

        scrollbar_v = ttk.Scrollbar(self.transition_matrix_frame, orient="vertical", command=canvas.yview)
        scrollbar_v.grid(row=0, column=2, sticky="ns")

        scrollbar_h = ttk.Scrollbar(self.transition_matrix_frame, orient="horizontal", command=canvas.xview)
        scrollbar_h.grid(row=1, column=0, columnspan=2, sticky="ew")

        matrix_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=matrix_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)

        # Display the grid
        for i, row in enumerate(self.transition_matrix):
            for j, value in enumerate(row):
                label = ttk.Label(matrix_frame, text=f"{value:.4f}", relief="solid", borderwidth=1)
                label.grid(row=i, column=j, padx=2, pady=2, sticky="nsew")

        # Adjust canvas scroll region
        matrix_frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))





    def toggle_uniform_probabilities(self):
        """Handle the uniform probabilities toggle."""
        if self.uniform_var.get():
            self.init_prob_entry.delete(0, tk.END)
            # Automatically set the entry to 1.0 divided by the number of states
            uniform_value = 1.0 / self.number_of_states if self.number_of_states else 0.0
            self.init_prob_entry.insert(0, str(uniform_value))
           



    def log_sum_exp(self, log_probs):
        """
        Computes the log of the sum of exponentials of input log probabilities.
        This avoids underflow/overflow issues with small probabilities.
        """
        max_log = max(log_probs)
        return max_log + math.log(sum(math.exp(lp - max_log) for lp in log_probs))


    def select_model(self):
        """
        Purpose:
            Select the chosen model and reveal the "Execute" button or additional controls for specific models.

        Inputs:
            None. Uses `self.fitModelChoice` and selected columns from comboboxes.

        Outputs:
            None. Reveals the "Execute" button or additional inputs.
        """
           # Check if there are any entry boxes currently visible
        entries_present = any(
            isinstance(widget, ttk.Entry) and widget.winfo_ismapped()
            for widget in self.processing_frame.winfo_children()
        )

        # Only clear entries if there are entries present
        if entries_present:
            self.clear_entries()


        if self.df is None:
            messagebox.showerror("Error", "Please load a DataFrame first!")
            return

        if not self.fitModelChoice:
            messagebox.showerror("Error", "Please select a model!")
            return

        column1 = self.column1_combobox.get()
        column2 = self.column2_combobox.get()

        if not column1 or not column2:
            messagebox.showerror("Error", "Please select both columns!")
            return

        def validate_sample_rate():
            try:
                self.sample_rate = float(self.sample_rate_entry.get())
                if self.sample_rate <= 0:
                    raise ValueError("Sample rate must be a positive non-zero float.")
                self.execute_button.config(state="normal")
                return True
            except ValueError:
                messagebox.showerror("Error", "Invalid sample rate. Must be a positive non-zero float.")
                return False

        def validate_max_chunk_size():
            try:
                self.max_chunk_size = int(self.max_chunk_size_entry.get())
                if self.max_chunk_size <= 0 or self.max_chunk_size >= len(self.df[column2].values):
                    raise ValueError("Max chunk size must be a positive integer less than the length of y_data.")
                self.execute_button.config(state="normal")
            except ValueError:
                messagebox.showerror("Error", "Invalid chunk size. Must be a positive integer less than the length of y_data.")


        def validate_changepoints():
            try:
                changepoint_str = self.changepoints_entry.get()
                changepoints = [int(i.strip()) for i in changepoint_str.split(',')]
                if any(cp < 0 or cp >= len(self.df[column2].values) for cp in changepoints):
                    raise ValueError("Changepoints must be positive integers and less than the length of y_data.")
                if sorted(changepoints) != changepoints:
                    raise ValueError("Changepoints must be in ascending order.")
                self.changepoints = changepoints
                return True
            except ValueError:
                messagebox.showerror("Error", "Invalid changepoint indices. Ensure they are positive integers, in ascending order, and within the data length.")
                return False

        def validate_rate_and_changepoints():
            if validate_sample_rate() and validate_changepoints():
                self.execute_button.config(state="normal")
            else:
                self.execute_button.config(state="disabled")

        # Below: Collect extra data depending on selected model:
        if self.fitModelChoice == 'Viterbi ---> 2-Stage Changepoint-Vector-State HMM':
            self.sample_rate_label = ttk.Label(self.processing_frame, text="Sample Rate (Hz)")
            self.sample_rate_label.grid(row=2, column=0, padx=5, pady=5)

            self.sample_rate_entry = ttk.Entry(self.processing_frame)
            self.sample_rate_entry.grid(row=2, column=1, padx=5, pady=5)

            self.changepoints_label = ttk.Label(self.processing_frame, text="Changepoint Indices")
            self.changepoints_label.grid(row=3, column=0, padx=5, pady=5)

            self.changepoints_entry = ttk.Entry(self.processing_frame)
            self.changepoints_entry.grid(row=3, column=1, padx=5, pady=5)

            self.confirm_changepoints_button = ttk.Button(self.processing_frame, text="Confirm", command=validate_rate_and_changepoints)
            self.confirm_changepoints_button.grid(row=4, column=1, padx=5, pady=5)

            self.execute_button = ttk.Button(self.processing_frame, text="Execute", command=self.execute_model, state="disabled")
            self.execute_button.grid(row=5, column=0, padx=5, pady=5)

        elif self.fitModelChoice in ['Viterbi --> Time-Dependent Pseudo-HMM (Weak)', 'Viterbi --> Time-Dependent Pseudo-HMM (Strong)', 'Viterbi --> HSMM']:
            self.sample_rate_label = ttk.Label(self.processing_frame, text="Sample Rate (Hz)")
            self.sample_rate_label.grid(row=2, column=0, padx=5, pady=5)

            self.sample_rate_entry = ttk.Entry(self.processing_frame)
            self.sample_rate_entry.grid(row=2, column=1, padx=5, pady=5)

            self.confirm_sample_rate_button = ttk.Button(self.processing_frame, text="Confirm", command=validate_sample_rate)
            self.confirm_sample_rate_button.grid(row=2, column=2, padx=5, pady=5)

            self.execute_button = ttk.Button(self.processing_frame, text="Execute", command=self.execute_model, state="disabled")
            self.execute_button.grid(row=3, column=0, padx=5, pady=5)

        elif self.fitModelChoice == 'Viterbi --> Averaged Chunked HMM':
            self.max_chunk_size_label = ttk.Label(self.processing_frame, text="Max Chunk Size")
            self.max_chunk_size_label.grid(row=2, column=0, padx=5, pady=5)

            self.max_chunk_size_entry = ttk.Entry(self.processing_frame)
            self.max_chunk_size_entry.grid(row=2, column=1, padx=5, pady=5)

            self.confirm_max_chunk_size_button = ttk.Button(self.processing_frame, text="Confirm", command=validate_max_chunk_size)
            self.confirm_max_chunk_size_button.grid(row=2, column=2, padx=5, pady=5)

            self.execute_button = ttk.Button(self.processing_frame, text="Execute", command=self.execute_model, state="disabled")
            self.execute_button.grid(row=3, column=0, padx=5, pady=5)

        else:
            self.execute_button = ttk.Button(self.processing_frame, text="Execute", command=self.execute_model)
            self.execute_button.grid(row=2, padx=5, pady=5)


    

    def execute_model(self):
        """
        Purpose:
            Execute the chosen model and display results.

        Inputs:
            None. Uses `self.fitModelChoice` and selected columns from comboboxes.

        Outputs:
            None. Updates the plot frame with the model's output.
        """
        try:
            column1 = self.column1_combobox.get()
            column2 = self.column2_combobox.get()

            x_data = self.df[column1].values
            y_data = self.df[column2].values

            model_out = self.run_model(y_data)
            fitted_y  = model_out['Emission Estimates']
            best_path = model_out['State Estimates']

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(x_data, y_data, label="Original Data", alpha=0.7)
            ax.plot(x_data, fitted_y, label="Decoded Model", color="red", linewidth=2)
            ax.set_xlabel(column1)
            ax.set_ylabel(column2)
            ax.set_title(f"Model Fit: {self.fitModelChoice}")
            ax.legend()
            ax.grid(True)

            self.plot_to_frame(fig)

            self.export_button = ttk.Button(self.processing_frame, text="Export Fit", command=lambda: self.export_fit(y_data, fitted_y, best_path))
            self.export_button.grid(row=9, column=0, padx=5, pady=5)
        except Exception as e:
            messagebox.showerror("Error", f"Error executing model: {e}")

        finally:
            self.clear_entries()

                

    def run_model(self, y_data):
        """
        Purpose:
            Simulate the execution of a model based on the selected model type.

        Inputs:
            model_name (str): The selected model name.
            x_data (ndarray): The x-axis data from the DataFrame.
            y_data (ndarray): The y-axis data from the DataFrame.

        Outputs:
            tuple: (fitted_x, fitted_y) where both are identical to inputs in this placeholder.
        """
        if self.fitModelChoice == 'Viterbi --> HMM':
            fitted_states, fitted_y = self.run_viterbi_hmm(y_data)
            return {'State Estimates' : fitted_states, 'Emission Estimates' : fitted_y}
       
        elif self.fitModelChoice =='Viterbi --> Time-Dependent Pseudo-HMM (Weak)':
            fitted_states, fitted_y = self.run_timedep_pseudohmm_weak(y_data, self.sample_rate)
            return {'State Estimates' : fitted_states, 'Emission Estimates' : fitted_y}
        
        elif self.fitModelChoice == 'Viterbi --> Time-Dependent Pseudo-HMM (Strong)':
            fitted_states, fitted_y = self.run_timedep_pseudohmm_strong(y_data, self.sample_rate)
            return {'State Estimates' : fitted_states, 'Emission Estimates' : fitted_y}
        
        elif self.fitModelChoice == 'Viterbi --> Averaged Chunked HMM':
            fitted_states, fitted_y = self.run_avg_chunked_hmm(y_data, self.max_chunk_size)
            return {'State Estimates' : fitted_states, 'Emission Estimates' : fitted_y}

        elif self.fitModelChoice == 'Viterbi --> HSMM':
            fitted_states, fitted_y = self.run_hsmm(y_data, self.sample_rate)
            return {'State Estimates' : fitted_states, 'Emission Estimates' : fitted_y}


        elif self.fitModelChoice == 'Forward-Backward --> HMM':
            fitted_states, fitted_y = self.run_fwd_bwd(y_data)
            return {'State Estimates' : fitted_states, 'Emission Estimates' : fitted_y}
        
        elif self.fitModelChoice == 'Viterbi ---> 2-Stage Changepoint-Vector-State HMM':
            fitted_states, fitted_y = self.run_2stage_changepoint_vector_state_hmm(y_data, self.changepoints, 1./self.sample_rate)
            return {'State Estimates' : fitted_states, 'Emission Estimates' : fitted_y}
            
        return 
    
    def export_fit(self, y_data, fitted_y, best_path):
        """
        Exports y_data and fitted_y as a dataframe chosen by the user using file explorer.
        """
        try:
            # Create the dataframe
            df_export = pd.DataFrame({
                'y_data': y_data,
                'fitted_y': fitted_y,
                'state estimate' : best_path
            })

            # Open the file explorer for saving
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")]
            )

            if file_path:
                df_export.to_csv(file_path, index=False)
                messagebox.showinfo("Success", "Data exported successfully!")
            else:
                messagebox.showwarning("Warning", "Export cancelled!")

        except Exception as e:
            messagebox.showerror("Error", f"Error exporting data: {e}")
    #################################################################################################################################################################################
    #################################################################################################################################################################################
    #################################################################################################################################################################################
    ####################################################### BELOW: PREPARATION FUNCTIONS FOR DECODER ALGORITHMS ####################################################################
    #################################################################################################################################################################################
    #################################################################################################################################################################################
    #################################################################################################################################################################################
    #################################################################################################################################################################################
    def confirm_sample_rate(self):
        try:
            rate = float(self.sample_rate_entry.get())
            if rate <= 0:
                raise ValueError("Sample rate must be a positive value.")
            self.sample_rate = rate
            messagebox.showinfo("Info", f"Sample rate set to {self.sample_rate} Hz")
        except ValueError as e:
            messagebox.showerror("Error", str(e))
    
    def confirm_changepoints(self):
        try:
            # Read and validate changepoints
            raw_changepoints = self.changepoints_entry.get().split(",")
            self.changepoints = [int(cp.strip()) for cp in raw_changepoints]

            # Ensure values are sorted and within the valid range
            if any(cp < 0 or cp >= len(self.df[self.column2_combobox.get()].values) for cp in self.changepoints):
                raise ValueError("Changepoint indices must be non-negative and less than the length of the dataset.")

            if sorted(self.changepoints) != self.changepoints:
                raise ValueError("Changepoint indices must be in ascending order.")

            messagebox.showinfo("Success", "Changepoints confirmed successfully!")

        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    

    def get_hmm_params(self):
        """Retrieve HMM parameters from the Model Frame for use in run_viterbi_hmm."""
        return self.initial_probabilities, self.state_means, self.state_vars, self.transition_matrix
    
    def get_hsmm_params(self):
        """Retrieve HSMM parameters from the Model Frame for use in run_viterbi_hsmm."""
        return self.initial_probabilities, self.state_means, self.state_vars, self.transition_matrix, self.hsmm_parameters["d"], self.hsmm_parameters["v"], self.hsmm_parameters["s"]
             



    def run_viterbi_hmm(self, y_data):
        start_prob, means, vars, transition_matrix = self.get_hmm_params()
        states                                    = [State(mean, var, ) for mean, var in zip(means, vars)]
        viterbi_hmm_out                           = self.log_viterbi_algorithm(y_data, states, start_prob, transition_matrix)
        best_path                                 = viterbi_hmm_out[0]
        emission_estimates                        = np.array([means[i] for i in best_path])
        return best_path, emission_estimates
    
    def run_fwd_bwd(self, y_data):
        start_prob, means, vars, transition_matrix = self.get_hmm_params()
        states                                    = [State(mean, vars, ) for mean, var in zip(means, vars)]
        fb_out                                    = self.forward_backward_log(y_data, states, np.log(start_prob), np.log(transition_matrix))

        best_states = fb_out[1]
        emission_estimates = np.array([means[i] for i in best_states])
        return best_states, emission_estimates

    def run_timedep_pseudohmm_strong(self, y_data, sample_rate):
        start_prob, means, vars, transition_matrix, d_params, v_params, s_params = self.get_hsmm_params()
        dvs_params                                                               = [[d,v,s] for d,v,s in zip(d_params, v_params, s_params)]

        states          = [State(mean, var, self.inverse_gauss_phys, dvs) for mean, var, dvs in zip(means, vars, dvs_params)]
        cdf_table       = []


        for s in states:
            dmax = self.get_ROI(states[0], 0.0, self.sample_rate, alpha_low=0.1, alpha_upp=0.9999)[1]
            cdf_table.append(self.get_cumulative_lookup(s.duration_dist, 0, sample_rate, dmax, s.duration_dist_params))

        viterbi_out = self.JHSMM_strong(y_data, states, start_prob, transition_matrix, cdf_table)

        best_path = viterbi_out[0]


        emission_estimates = np.array([means[i] for i in best_path])
        return best_path, emission_estimates

    
    def run_timedep_pseudohmm_weak(self, y_data, sample_rate):
        start_prob, means, vars, transition_matrix, d_params, v_params, s_params = self.get_hsmm_params()
        dvs_params                                                               = [[d,v,s] for d,v,s in zip(d_params, v_params, s_params)]

        states          = [State(mean, var, self.inverse_gauss_phys, dvs) for mean, var, dvs in zip(means, vars, dvs_params)]
        cdf_table       = []


        for s in states:
            dmax = self.get_ROI(states[0], 0.0, self.sample_rate, alpha_low=0.1, alpha_upp=0.9999)[1]
            cdf_table.append(self.get_cumulative_lookup(s.duration_dist, 0, sample_rate, dmax, s.duration_dist_params))

        viterbi_out = self.JHSMM_weak(y_data, states, start_prob, transition_matrix, cdf_table)

        best_path = viterbi_out[0]


        emission_estimates = np.array([means[i] for i in best_path])
        return best_path, emission_estimates
    def run_hsmm(self, y_data, sample_rate):
        start_prob, means, vars, transition_matrix, d_params, v_params, s_params = self.get_hsmm_params()
        dvs_params                                                               = [[d,v,s] for d,v,s in zip(d_params, v_params, s_params)]

        states          = [State(mean, var, self.inverse_gauss_phys, dvs) for mean, var, dvs in zip(means, vars, dvs_params)]
        cdf_table       = []


        for s in states:
            dmax = self.get_ROI(states[0], 0.0, self.sample_rate, alpha_low=0.1, alpha_upp=0.9999)[1]
            cdf_table.append(self.get_cumulative_lookup(s.duration_dist, 0, sample_rate, dmax, s.duration_dist_params))



        # Cutting off extreme values to reduce computational complexity:
        ROI          = [self.get_ROI(S, 0.0, sample_rate, 0.005, 0.999) for S in states]
        max_duration = np.array([x[1] for x in ROI])
        min_duration = np.array([x[0] for x in ROI])


        viterbi_out = self.viterbi_hsmm(y_data, states, start_prob, transition_matrix, min_duration, max_duration, sample_rate, cdf_table)

        # Below: viterbi_hsmm outputs states as (state index, state duration). Therefore, we need to "unfold" the states such that the state notation of (state index, state duration),
        # instead becomes "state index" repeated "state duration" times in a row. 
        best_state    = [x[0] for x in viterbi_out[0]]
        best_duration = [x[1] for x in viterbi_out[0]]

        states_pointwise = []

        num_states_est = len(best_state)
        for i in range(num_states_est):
            states_pointwise.append(np.repeat(best_state[i], best_duration[i]))
            
        states_pointwise = np.concatenate(states_pointwise)

        best_path          = states_pointwise
        emission_estimates = np.array([means[x] for x in best_path]) 

        return best_path, emission_estimates

    def run_avg_chunked_hmm(self, y_data, max_chunk_size):
        start_prob, means, vars, transition_matrix = self.get_hmm_params()
        states                                    = [State(mean, var, ) for mean, var in zip(means, vars)]
        


        best_path = self.loop_chunk_hmm(y_data, max_chunk_size, states, start_prob, transition_matrix)

        emission_estimates                        = np.array([means[int(i)] for i in best_path])
        return best_path, emission_estimates

    def chunks_to_pointwise(self, observations, L, most_probable_chunks):
        from math import floor
        num_obs      = len(observations)
        pointwise    = np.zeros(num_obs, dtype=int)
        num_chunks   = floor(num_obs/L)
        num_leftover = int(num_obs - num_chunks*L)
        for i in range(num_chunks):
            start_index = i*L
            end_index   = (i+1)*L
            
            pointwise[start_index:end_index] = most_probable_chunks[i]
        
        if num_leftover > 0:
            emission_index = num_chunks*L
            for i in range(num_chunks, num_chunks+num_leftover):
                pointwise[emission_index] = most_probable_chunks[i]
                emission_index += 1
        return pointwise

    def loop_chunk_hmm(self, observations, num_L, states, start_prob, transition_matrix):
        # Trying an average of the predictions over many chunk sizes:
  
        num_obs = len(observations)
        all_pointwise = []
        for L in range(1, num_L+1):
            chunked_out  = self.log_viterbi_algo_chunks(observations, states, start_prob, transition_matrix, L)
            chunked_path = chunked_out[0]
            
            pointwise = self.chunks_to_pointwise(observations, L, chunked_path)
            all_pointwise.append(pointwise)
            
        avg_pointwise = np.zeros(num_obs)
        var_pointwise = np.zeros(num_obs)
        for i in range(num_obs):
            avg_pointwise[i] = np.round(np.mean([x[i] for x in all_pointwise]))
            

        return avg_pointwise


    def run_fwdbwd_hmm(self, y_data):
        start_prob, means, vars, transition_matrix = self.get_hmm_params()
        states                                     = [State(mean, var, ) for mean, var in zip(means, vars)]
        
        fwdbwd_hmm_out                             = self.forward_backward_log(y_data, states, np.log(start_prob), np.log(transition_matrix))
        best_path                                  = fwdbwd_hmm_out[1]
        emission_estimates                         = np.array([means[i] for i in best_path])
        return best_path, emission_estimates
    

    def run_2stage_changepoint_vector_state_hmm(self, y_data, state_changepoints, gamma):
        start_prob, means, vars, transition_matrix, d_params, v_params, s_params = self.get_hsmm_params()
        dvs_params                                                               = [[d,v,s] for d,v,s in zip(d_params, v_params, s_params)]
        states          = [State(mean, var, self.inverse_gauss_phys, dvs) for mean, var, dvs in zip(means, vars, dvs_params)]


        def split_time_series(time_series, changepoints):
            """
            Splits a time series array into sub-arrays using specified changepoint indices.

            Args:
                time_series (np.ndarray): The input time series array.
                changepoints (list or np.ndarray): Array of changepoint indices.

            Returns:
                list of np.ndarray: A list of arrays, each representing a segment of the time series.
            """
            if not isinstance(time_series, np.ndarray):
                raise TypeError("time_series must be a numpy array")
            if not isinstance(changepoints, (list, np.ndarray)):
                raise TypeError("changepoints must be a list or numpy array")
            if any(cp < 0 or cp >= len(time_series) for cp in changepoints):
                raise ValueError("Changepoints must be within the range of the time series")

            # Ensure changepoints are sorted
            #changepoints = sorted(changepoints)
            # Add the end of the series to the changepoints for final split
            changepoints = changepoints + [len(time_series)]
            changepoints.insert(0, 0)


            # Generate sub-arrays
            segments = [
                time_series[changepoints[i]:changepoints[i + 1]].ravel()
                for i in range(len(changepoints) - 1)
            ]
            return segments


        segments = split_time_series(y_data, state_changepoints)

        vector_hmm_out = self.viterbi_algorithm_vector(segments, states, start_prob, transition_matrix, gamma, tol=1e-8)

        best_path = vector_hmm_out[0]
        best_path_pointwise = []
        num_segments = len(segments)
        for i in range(num_segments):
            best_path_pointwise.append(np.repeat(best_path[i], len(segments[i]))) 
        
        best_path_pointwise = np.concatenate(best_path_pointwise)
        emission_estimates = np.array([means[int(x)] for x in best_path_pointwise])

        return best_path, emission_estimates

    def collect_initial_probabilities(self):
        # Step 2: Add label and entry box for initial probabilities
        self.init_prob_label = ttk.Label(self.processing_frame, text="Initial Probabilities:")
        self.init_prob_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")

        self.current_state = 0  # To track the state for which probabilities are being entered
        self.init_probs = []  # To store the probabilities entered by the user

        self.init_prob_state_label = ttk.Label(
            self.processing_frame, text=f"Initial Probability of State {self.current_state}:"
        )
        self.init_prob_state_label.grid(row=5, column=0, padx=5, pady=5, sticky="w")

        self.init_prob_entry = ttk.Entry(self.processing_frame)
        self.init_prob_entry.grid(row=5, column=1, padx=5, pady=5, sticky="w")

        # Add confirm button for probabilities
        def confirm_initial_prob():
            try:
                prob = float(self.init_prob_entry.get())
                if prob < 0 or prob > 1:
                    raise ValueError("Probability must be between 0 and 1.")

                if self.uniform_var.get():
                    # Use the same probability for all states
                    self.init_probs = [prob] * self.num_states
                    if abs(sum(self.init_probs) - 1.0) > 1e-6:
                        messagebox.showerror(
                            "Error", "Probabilities do not sum to 1. Please re-enter."
                        )
                    else:
                        messagebox.showinfo(
                            "Success", "Uniform initial probabilities set successfully!"
                        )
                        self.next_steps_after_probabilities()
                else:
                    # Store the entered probability and proceed to the next state
                    self.init_probs.append(prob)
                    self.current_state += 1
                    if self.current_state < self.num_states:
                        self.init_prob_state_label.config(
                            text=f"Initial Probability of State {self.current_state}:"
                        )
                        self.init_prob_entry.delete(0, tk.END)
                    else:
                        # All probabilities collected; validate sum
                        if abs(sum(self.init_probs) - 1.0) > 1e-6:
                            messagebox.showerror(
                                "Error", "Probabilities do not sum to 1. Please re-enter."
                            )
                            self.init_probs = []
                            self.current_state = 0
                            self.init_prob_state_label.config(
                                text=f"Initial Probability of State {self.current_state}:"
                            )
                            self.init_prob_entry.delete(0, tk.END)
                        else:
                            messagebox.showinfo(
                                "Success", "Initial probabilities entered successfully!"
                            )
                            self.next_steps_after_probabilities()

            except ValueError as e:
                messagebox.showerror("Error", f"Invalid probability: {e}")

        self.init_prob_confirm_button = ttk.Button(
            self.processing_frame, text="Confirm Probability", command=confirm_initial_prob
        )
        self.init_prob_confirm_button.grid(row=5, column=2, padx=5, pady=5, sticky="w")


    def next_steps_after_probabilities(self):
        # Placeholder for the next steps after probabilities are set
        messagebox.showinfo("Next Steps", "Proceeding to the next steps of the model.")


    #################################################################################################################################################################################
    #################################################################################################################################################################################
    #################################################################################################################################################################################
    ###################################################### BELOW: HELPER FUNCTIONS FOR DECODER ALGORITHMS ###########################################################################
    #################################################################################################################################################################################
    #################################################################################################################################################################################
    #################################################################################################################################################################################
    #################################################################################################################################################################################
    def get_best_duration_index(self, valid_states, best_duration):
        '''
        Purpose: The objective of this function is to return the index of the entry in valid_states for which the entry is equal to best_duration
                 The is required in the hsmm viterbi algorithm. 

        Input:
            - valid_states (list of integers) - Entries correspond to values of discretized duration in the hsmm Viterbi algorithm
            - best_duration (int)             - the duration parameter of the best previous state as determined by the HSMM Viterbi algorithm

        Output:
         - i (int) - when best_duration matches one of the entries of valid_states, we return the corresponding index, i, of valid_states
        '''
        num_valid_states = len(valid_states)
        for i in range(num_valid_states):
            if best_duration == valid_states[i]:
                return i
        
        return None

    def get_cumulative_probability(self, pdf, low_bound, upp_bound, args):
        '''
        Purpose: Return the cumulative probability of some continuous probability distribution function, pdf. 
                 This is required for computing transition probabilities in all models beyond the basic HMM.

        Input:
            - pdf (function)    : a probability distribution function
            - low_bound (float) : the lower limit in the definite integral over the pdf
            - upp_bound (float) : the upper limit in the definite integral over the pdf
        
        Output:
         - result (float)       : the cumulative probability of the pdf between low_bound and upp_bound
        '''
        from scipy.integrate import quad
        if args:
            result, error = quad(pdf, low_bound, upp_bound, tuple(args))
        else:
            result, error =  quad(pdf, low_bound, upp_bound)
        return result

    def get_low_lim(self, tcut, thresh, f, alpha, args):
        '''
        Purpose: 
                This function returns the difference between a specified value, alpha, and the integral 
                a probability distribution function between the values of thresh (lower limit) and tcut
                (upper limit).
                This function is intended to be use as a loss function where we solve for tcut, given a
                 pdf and associated alpha value, where thresh is then  the minimum of the domain of the 
                 pdf (often -inf, or zero .... but not always!).
                 The solution is the value of tcut, or cut-off value, which defines the lower "tail"
                 of the pdf, such that the probability of a value below tcut is equal to alpha. 

        Input:
            - tcut (float)   : upper limit of the integral over f
            - f (function)   : a probability distribution function
            - thresh (float) : lower limit of the integral over f
            - alpha (float)  : target value of the integral over f, representing the probability contained between
                               thresh and tcut
            - args (list)    : any argumenty/parameters required to specify f
        Output:

         
        '''
        return self.get_cumulative_probability(f, thresh, tcut, args) - alpha

    def get_upp_lim(self, tcut, thresh, f, alpha, args):
        return self.get_cumulative_probability(f, thresh, tcut, args) - alpha
    
    def get_ROI(self, S, thresh, sample_rate, alpha_low=0.1, alpha_upp=0.9):
        import numpy as np
        from math import floor, ceil
        from scipy.optimize import fsolve
        x0=S.duration_dist_params[0]/S.duration_dist_params[1]
        low_cut = fsolve(self.get_low_lim, x0 = x0, args = (thresh, S.duration_dist, alpha_low, S.duration_dist_params))
        upp_cut = fsolve(self.get_upp_lim, x0 = x0, args = (thresh, S.duration_dist, alpha_upp, S.duration_dist_params))

        nlow = floor(low_cut*sample_rate)
        nupp = ceil(upp_cut*sample_rate)
        
        ROI = [nlow, nupp]
        
        return ROI

    def get_cumulative_lookup(self, pdf, min_bound, sample_rate, dmax, args):
        '''
        Purpose: To create a table of values for a CDF sampled at a fixed step. Entry d corresponds to the CDF evaluated at sample_rate*d
        '''
        lookup_table = np.zeros(dmax)
        
        lookup_table = np.array([self.get_cumulative_probability(pdf, min_bound, d/sample_rate, args) for d in (np.arange(dmax) + 1)])
        
        return lookup_table
   
    def inverse_gauss_phys(self, t, d, v, s):
        '''
        Parameters:
            d : pore length
            v : classical drift speed in medium || depends on model... could be a mobility thing in an E field, could be something else
            s : volatility component which scales the noise parameter in the model || units are distance/time^-0.5
        '''
        import numpy as np
        from math import pi as pi
        coef = d/s/np.sqrt(2*pi*t**3)
        arg  = (v*t - d)**2 / (2 * s**2 * t)
        
        density = coef*np.exp(-arg)
        
        return density

    #################################################################################################################################################################################
    #################################################################################################################################################################################
    #################################################################################################################################################################################
    ###################################################### BELOW: FUNCTION DEFINITIONS OF VARIOUS AVAILABLE DECODER ALGORITHMS ######################################################
    #################################################################################################################################################################################
    #################################################################################################################################################################################
    #################################################################################################################################################################################
    #################################################################################################################################################################################
    def log_viterbi_algorithm(self, obs, states, start_prob, transition_matrix):
        # Step 1: Initialize Variables
        num_states        = len(states)
        num_obs           = len(obs)
        log_viterbi_table = [[0.0 for _ in range(num_states)] for _ in range(num_obs)]
        backpointer       = [[0 for _ in range(num_states)] for _ in range(num_obs)]

        # Step 2: Calculate Probabilities
        for t in range(num_obs):
            for s in range(num_states):
                if t == 0:
                    log_viterbi_table[t][s] = np.log(start_prob[s]) + np.log(states[s].emission_probability(obs[t]))
                else:
                    log_max_prob            = max(log_viterbi_table[t-1][prev_s] + np.log(transition_matrix[prev_s][s]) for prev_s in range(num_states))
                    log_viterbi_table[t][s] = log_max_prob + np.log(states[s].emission_probability(obs[t]))
                    backpointer[t][s]       = max(range(num_states), key=lambda prev_s: log_viterbi_table[t-1][prev_s] + np.log(transition_matrix[prev_s][s]))

        # Step 3: Traceback and Find Best Path
        best_path_prob    = max(log_viterbi_table[-1])
        best_path_pointer = log_viterbi_table[-1].index(best_path_prob)
        #best_path_pointer = max(range(num_states), key=lambda s: viterbi_table[-1][s])
        best_path         = [best_path_pointer]
        for t in range(len(obs)-1, 0, -1):
            best_path.insert(0, backpointer[t][best_path[0]])

        # Step 4: Return Best Path and other info
        return best_path, backpointer, log_viterbi_table
    

    def viterbi_algorithm_vector(self, obs, states, start_prob, transition_matrix, gamma, tol=1e-8):
        # Step 2: Initialize Variables
        num_states        = len(states)
        num_obs           = len(obs)
        log_viterbi_table = [[0.0 for _ in range(num_states)] for _ in range(num_obs)]
        backpointer       = [[0 for _ in range(num_states)] for _ in range(num_obs)]
        # Step 3: Calculate Probabilities
        for t in range(num_obs):
            for s in range(num_states):
                if t == 0:
                    log_viterbi_table[t][s] = np.log(start_prob[s]) + states[s].log_joint_n_xvec(obs[t], gamma, tol=tol)
                else:
                    log_max_prob            = max(log_viterbi_table[t-1][prev_s] + np.log(transition_matrix[prev_s][s]) for prev_s in range(num_states))
                    log_viterbi_table[t][s] = log_max_prob + states[s].log_joint_n_xvec(obs[t], gamma, tol=tol)
                    backpointer[t][s]       = max(range(num_states), key=lambda prev_s: log_viterbi_table[t-1][prev_s] +  np.log(transition_matrix[prev_s][s]))
        # Step 4: Traceback and Find Best Path
        best_path_prob    = max(log_viterbi_table[-1])
        best_path_pointer = max(range(num_states), key=lambda s: log_viterbi_table[-1][s])
        best_path         = [best_path_pointer]
        for t in range(len(obs)-1, 0, -1):
            best_path.insert(0, backpointer[t][best_path[0]])
        # Step 5: Return Best Path
        return best_path, backpointer, log_viterbi_table
    
    def log_viterbi_algo_chunks(self, obs, states, start_prob, transition_matrix, L):
        from math import floor
        # Step 1: Initialize Variables
        num_states        = len(states)
        num_obs           = len(obs)
        num_chunks        = floor(num_obs/L)
        num_leftover      = int(num_obs - L * num_chunks) 
        log_viterbi_table = [[0.0 for _ in range(num_states)] for _ in range(num_chunks+num_leftover)]
        backpointer       = [[0 for _ in range(num_states)] for _ in range(num_chunks+num_leftover)]

        # Step 2: Calculate Probabilities
        
        for t in range(num_chunks):
            start_index = t * L
            end_index   = start_index + L
            for s in range(num_states):
                if t == 0:
                    log_viterbi_table[t][s] = np.log(start_prob[s]) + np.sum(np.array([np.log(states[s].emission_probability(obs[i])) for i in range(start_index, end_index)]))
                else:
                    log_max_prob            = max(log_viterbi_table[t-1][prev_s] + np.log(transition_matrix[prev_s][s]) for prev_s in range(num_states))
                    log_viterbi_table[t][s] = log_max_prob + np.sum(np.array([np.log(states[s].emission_probability(obs[i])) for i in range(start_index, end_index)]))
                    backpointer[t][s]       = max(range(num_states), key=lambda prev_s: log_viterbi_table[t-1][prev_s] + np.log(transition_matrix[prev_s][s]))

        if num_leftover > 0:
            emission_index = num_chunks*L 
            for t in range(num_chunks, num_chunks + num_leftover):
                for s in range(num_states):
                    log_max_prob            = max(log_viterbi_table[t-1][prev_s] + np.log(transition_matrix[prev_s][s]) for prev_s in range(num_states))
                    log_viterbi_table[t][s] = log_max_prob + np.log(states[s].emission_probability(obs[emission_index]))
                    backpointer[t][s]       = max(range(num_states), key=lambda prev_s: log_viterbi_table[t-1][prev_s] + np.log(transition_matrix[prev_s][s]))
                emission_index+=1
        # Step 3: Traceback and Find Best Path
        best_path_prob    = max(log_viterbi_table[-1])
        best_path_pointer = log_viterbi_table[-1].index(best_path_prob)
        #best_path_pointer = max(range(num_states), key=lambda s: viterbi_table[-1][s])
        best_path         = [best_path_pointer]
        for t in range(num_chunks+num_leftover-1, 0, -1):
            best_path.insert(0, backpointer[t][best_path[0]])

        # Step 4: Return Best Path and other info
        return best_path, backpointer, log_viterbi_table

    def forward_backward_log(self, obs, states, start_log_prob, transition_log_matrix):
        # Step 1: Initialize Variables
        num_states         = len(states)
        num_obs            = len(obs)
        forward_log_table  = [[-math.inf for _ in range(num_states)] for _ in range(num_obs)]
        backward_log_table = [[-math.inf for _ in range(num_states)] for _ in range(num_obs)]

        # Step 2: Forward Pass (Initialization)
        for s in range(num_states):
            forward_log_table[0][s] = start_log_prob[s] + math.log(states[s].emission_probability(obs[0]))

        # Step 3: Forward Pass (Recursion)
        for t in range(1, num_obs):
            for s in range(num_states):
                log_probs = [
                    forward_log_table[t - 1][prev_s] + transition_log_matrix[prev_s][s]
                    for prev_s in range(num_states)
                ]
                forward_log_table[t][s] = self.log_sum_exp(log_probs) + math.log(states[s].emission_probability(obs[t]))

        # Step 4: Backward Pass (Initialization)
        for s in range(num_states):
            backward_log_table[-1][s] = 0.0  # log(1) = 0

        # Step 5: Backward Pass (Recursion)
        for t in range(num_obs - 2, -1, -1):
            for s in range(num_states):
                log_probs = [
                    backward_log_table[t + 1][next_s]
                    + transition_log_matrix[s][next_s]
                    + math.log(states[next_s].emission_probability(obs[t + 1]))
                    for next_s in range(num_states)
                ]
                backward_log_table[t][s] = self.log_sum_exp(log_probs)

        # Step 6: Posterior Probabilities
        posterior_log_probs = [
            [forward_log_table[t][s] + backward_log_table[t][s] for s in range(num_states)]
            for t in range(num_obs)
        ]

        # Normalize posterior probabilities in log-space
        posterior_probs = []
        for t in range(num_obs):
            log_norm_factor = self.log_sum_exp(posterior_log_probs[t])
            posterior_probs.append([math.exp(lp - log_norm_factor) for lp in posterior_log_probs[t]])

        # Step 7: Most Likely State at Each Time Step
        most_likely_states = [max(range(num_states), key=lambda s: posterior_probs[t][s]) for t in range(num_obs)]

        return posterior_probs, most_likely_states, forward_log_table, backward_log_table

    def log_sum_exp_logs(self, log_a, log_b):
        """
        Compute log(exp(log_a) + exp(log_b)) in a numerically stable way.
        """
        M = max(log_a, log_b)
        return M + np.log(np.exp(log_a - M) + np.exp(log_b - M))


    def JHSMM_weak(self, observations, states, start_prob, trans_prob, cdf_table):
        """
        Viterbi algorithm with tracking of consecutive state occurrences.

        Parameters:
            observations: list of observations
            states: list of possible states (objects with an emission_probability method)
            start_prob: dict of starting probabilities for each state
            trans_prob: 2D list, transition probabilities between states (indexed by state indices)

        Returns:
            tuple:
                - list of the most likely states (optimal path)
                - float, the log probability of the optimal path
                - 2D list of running counts of consecutive occurrences for each state at each time step
        """
        n_obs      = len(observations)
        num_states = len(states)
        
        # Initialize dynamic programming tables
        viterbi_probs      = np.full((n_obs, num_states), -np.inf)  # Log probabilities of paths
        back_pointers      = np.zeros((n_obs, num_states), dtype=int)  # Backtracking table
        consecutive_counts = np.zeros((n_obs, num_states), dtype=int)  # Running counts of consecutive occurrences
        
        # Initialization step (t = 0)
        for s in range(num_states):
            viterbi_probs[0, s]      = np.log(start_prob[s]) + np.log(states[s].emission_probability(observations[0]))
            back_pointers[0, s]      = -1  # No previous state at t=0
            consecutive_counts[0, s] = 1  # First occurrence of each state

        # Iteration step (t = 1 to n_obs - 1)
        for t in range(1, n_obs):
            for s in range(num_states):  # Current state
                max_prob               = -np.inf
                best_prev_state        = -1
                
                for prev_s in range(num_states):  # Previous state
                    d_prevailing = consecutive_counts[t-1][prev_s]
                    if s != prev_s:
                        if d_prevailing >= len(cdf_table[prev_s]):
                            duration_factor = 1
                        else:
                            duration_factor = cdf_table[prev_s][d_prevailing]#get_cumulative_probability(states[prev_s].duration_dist, 0.00001, consecutive_counts[t-1, prev_s]/sample_rate, states[prev_s].duration_dist_params)
                        prob = viterbi_probs[t-1, prev_s] + np.log(trans_prob[prev_s][s]) + np.log(duration_factor)
                    else:
                        prob          = viterbi_probs[t-1, prev_s] + np.log(trans_prob[prev_s][s]) 
                    
                    if prob > max_prob:
                        max_prob        = prob
                        best_prev_state = prev_s

                # Update running count based on the best previous state
                if best_prev_state == s:
                    best_consecutive_count = consecutive_counts[t-1, best_prev_state] + 1
                else:
                    best_consecutive_count = 1
                # Update tables
                viterbi_probs[t, s]      = max_prob + np.log(states[s].emission_probability(observations[t]))
                back_pointers[t, s]      = best_prev_state
                consecutive_counts[t, s] = best_consecutive_count # Default to 1 if no consecutive match

        # Termination step: Find the most likely final state
        last_state = np.argmax(viterbi_probs[-1])
        log_prob   = viterbi_probs[-1, last_state]
        
        # Backtrack to find the optimal path
        optimal_path = [last_state]
        for t in range(n_obs - 1, 0, -1):
            last_state = back_pointers[t, last_state]
            optimal_path.append(last_state)
        
        optimal_path.reverse()  # Reverse to get the path from start to end
        return optimal_path, log_prob, consecutive_counts




    def JHSMM_strong(self, observations, states, start_prob, trans_prob, cdf_table):
        """
        Viterbi algorithm with tracking of consecutive state occurrences.

        Parameters:
            observations: list of observations
            states: list of possible states (objects with an emission_probability method)
            start_prob: dict of starting probabilities for each state
            trans_prob: 2D list, transition probabilities between states (indexed by state indices)

        Returns:
            tuple:
                - list of the most likely states (optimal path)
                - float, the log probability of the optimal path
                - 2D list of running counts of consecutive occurrences for each state at each time step
        """
        n_obs      = len(observations)
        num_states = len(states)
        
        # Initialize dynamic programming tables
        viterbi_probs      = np.full((n_obs, num_states), -np.inf)  # Log probabilities of paths
        back_pointers      = np.zeros((n_obs, num_states), dtype=int)  # Backtracking table
        consecutive_counts = np.zeros((n_obs, num_states), dtype=int)  # Running counts of consecutive occurrences
        
        # Initialization step (t = 0)
        for s in range(num_states):
            viterbi_probs[0, s]      = np.log(start_prob[s]) + np.log(states[s].emission_probability(observations[0]))
            back_pointers[0, s]      = -1  # No previous state at t=0
            consecutive_counts[0, s] = 1  # First occurrence of each state

        # Iteration step (t = 1 to n_obs - 1)
        for t in range(1, n_obs):
            for s in range(num_states):  # Current state
                max_prob               = -np.inf
                best_prev_state        = -1
                
                for prev_s in range(num_states):  # Previous state
                    d_prevailing = consecutive_counts[t-1][prev_s]
                    if s != prev_s:
                        if d_prevailing >= len(cdf_table[prev_s]):
                            duration_factor = 1
                        else:
                            duration_factor = cdf_table[prev_s][d_prevailing]#get_cumulative_probability(states[prev_s].duration_dist, 0.00001, consecutive_counts[t-1, prev_s]/sample_rate, states[prev_s].duration_dist_params)
                        prob = viterbi_probs[t-1, prev_s] + np.log(trans_prob[prev_s][s]) + np.log(duration_factor)
                    else:
                        if d_prevailing < len(cdf_table[prev_s]):
                            log1            = np.log(1. - trans_prob[prev_s][s]) + np.log(cdf_table[prev_s][d_prevailing])
                            update_factor   = 1. - np.exp(log1)
                            prob            = viterbi_probs[t-1, prev_s] + np.log(update_factor) #viterbi_probs[t-1, prev_s] + logsumexp_out
                        else:
                            update_factor   = trans_prob[prev_s][s] 
                            prob            = viterbi_probs[t-1, prev_s] + np.log(update_factor) #viterbi_probs[t-1, prev_s] + logsumexp_out
                    if prob > max_prob:
                        max_prob        = prob
                        best_prev_state = prev_s

                # Update running count based on the best previous state
                if best_prev_state == s:
                    best_consecutive_count = consecutive_counts[t-1, best_prev_state] + 1
                else:
                    best_consecutive_count = 1
                # Update tables
                viterbi_probs[t, s]      = max_prob + np.log(states[s].emission_probability(observations[t]))
                back_pointers[t, s]      = best_prev_state
                consecutive_counts[t, s] = best_consecutive_count # Default to 1 if no consecutive match

        # Termination step: Find the most likely final state
        last_state = np.argmax(viterbi_probs[-1])
        log_prob   = viterbi_probs[-1, last_state]
        
        # Backtrack to find the optimal path
        optimal_path = [last_state]
        for t in range(n_obs - 1, 0, -1):
            last_state = back_pointers[t, last_state]
            optimal_path.append(last_state)
        
        optimal_path.reverse()  # Reverse to get the path from start to end
        return optimal_path, log_prob, consecutive_counts

    def viterbi_hsmm(self, obs, states, start_prob, transition_matrix, min_duration, max_duration, sample_rate, cdf_lookup):
        num_states = len(states)
        num_obs    = len(obs)

        #min_duration = np.repeat(2, len(states))
        # Log-probability tables
        log_viterbi_table = []
        valid_state_table = []
        backpointer       = [] # [[np.array([None, None]) for _ in range(num_states)] for _ in range(num_obs)]  # Stores [prev_state, duration]

        # Log of the transition matrix:
        log_transition_matrix = np.log(transition_matrix)

        # Recurrence (includes initialization logic for t = 0)
        for t in range(num_obs):
            log_viterbi_table.append([])
            valid_state_table.append([])
            backpointer.append([])
            for s in range(num_states):
                log_viterbi_table[t].append([])
                valid_state_table[t].append([])
                backpointer[t].append([])
                # Initialize log of the joint emission probability for recursive calculation. We pre-calculate up to the minimum duration minus one.
                # Why minus one? Because it allows us to add a single new emission to the sum as we loop over valid durations beginning at the minimum
                # duration. If we did not include the "minus one", then the minimum duration would be a special case and we would require an if-statement
                # to check the value of duration at each iteration in the loop. 
                
                    
                
                if t > 0:
                    log_emission_prob  = np.log(states[s].emission_probability(obs[t])) 
                    log_emission_prob  += np.sum(np.log(states[s].emission_probability(obs[t-min_duration[s]+2:t]))) # Note that this will "wrap" if min_duration[s] > t... it doesn't matter though because, in such a scenario, we break the loop further ahead
                else:
                    log_emission_prob = 0
                
                # Loop over possible values of duration, d. Note that d being within a state's maximum and minimum values does not guarantee
                # that it is a valid duration. It must also either be the first state in the sequence or have a valid prior state. To have
                # a valid prior state, the duration must be such that it leaves a sufficient number of prior points, where the number of prior 
                # points is sufficient if it is greater than or equal to the minimum duration of at least one state class.
                for d_index in range(max_duration[s] - min_duration[s] + 1):
                    d = d_index + min_duration[s]
                    if t - (d - 1) < 0:  # Ensure valid duration
                        break

                    log_emission_prob += np.log(states[s].emission_probability(obs[t-d+1]))
                    log_duration_prob  = np.log(states[s].prob_n(d, sample_rate))
                    if t - (d - 1) == 0:  # Initialization case
                        # Calculate emission probability recursively      
                        if t == 0:
                            log_emission_prob  = np.log(states[s].emission_probability(obs[t]))  
                        else:
                            log_emission_prob  = np.log(states[s].emission_probability(obs[t])) 
                            log_emission_prob  += np.sum(np.log(states[s].emission_probability(obs[:t])))
                        log_prob           = np.log(start_prob[s]) + log_duration_prob + log_emission_prob
                        log_viterbi_table[t][s].append(log_prob)
                        valid_state_table[t][s].append(d)
                        backpointer[t][s].append(np.array([None, None]))
                    else:  # General case where t-(d-1) > 0 --> that is, we are looking for valid states that are not the first state in the sequence
                        # List of transition probabilities along with prev_s and prev_d
                        transition_probs = []
                        for prev_s in range(num_states):
                            if valid_state_table[t-d][prev_s]:
                                #if t-(d-1) >= min(valid_state_table[t-d][prev_s]):#(min_duration[prev_s]-1):
                                for prev_d_index in range(len(valid_state_table[t-d][prev_s])):
                                            
                                    # Calculate emission probability recursively
                                    prev_d = valid_state_table[t-d][prev_s][prev_d_index]
                                    transition_probs.append( (np.array(log_viterbi_table[t - d][prev_s][prev_d_index]) + log_transition_matrix[prev_s][s] + np.log(cdf_lookup[prev_s][prev_d]) , prev_s, prev_d))

                        if transition_probs:  # Check if transitions exist
                            # Find the index of the maximum transition probability
                            max_tuple           = max(transition_probs, key=lambda x: x[0])
                            max_transition_prob = max_tuple[0]
                            prev_s              = max_tuple[1] 
                            prev_d              = max_tuple[2]
                            
                            # Update log-probability table and valid state table:
                            log_prob           = max_transition_prob  + log_emission_prob + log_duration_prob #+ np.log(1.-cdf_lookup[s][d]) 
                            log_viterbi_table[t][s].append(log_prob)
                            valid_state_table[t][s].append(d)
                            backpointer[t][s].append(np.array([prev_s, prev_d]))

                    
                

        # Traceback
        best_path = []
        best_state, best_duration_index = max(
                                        ((s, d_index) for s in range(num_states) for d_index in range(len(log_viterbi_table[-1][s]))),
                                        key=lambda x: log_viterbi_table[-1][x[0]][x[1]]
                                        )

        best_duration = valid_state_table[-1][best_state][best_duration_index]
        best_path.append((best_state, best_duration))

        t                = num_obs - 1
        current_duration = best_duration
        while t - current_duration >= 0:
            prev_state, prev_duration = backpointer[t][best_state][best_duration_index]

            best_path.insert(0, (prev_state, prev_duration))
            prev_duration_index = self.get_best_duration_index(valid_state_table[t-current_duration][prev_state], prev_duration)
            
            t                  -= current_duration
            current_duration    = prev_duration
            best_state          = prev_state
            best_duration_index = prev_duration_index
            
            if current_duration == None:
                break
        return best_path, backpointer, log_viterbi_table, valid_state_table



def main():
    """
    Purpose:
        Create the root Tkinter window and initialize the DataFramePlotterApp.
    
    Inputs:
        None. Initializes the Tkinter environment.

    Outputs:
        None. Runs the main event loop for the application.
    """
    root = tk.Tk()
    app = DataFramePlotterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
