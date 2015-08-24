import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Analyzer(object):
    def __init__(self):
        super(Analyzer, self).__init__()
        if not os.path.isdir("tests"):
            raise Exception("Directory tests/ doesn't exist. Are you running from the top level?")

        filepath = os.path.join("tests", "times.csv")
        if not os.path.isfile(filepath):
            raise Exception("tests/times.csv doesn't exist.")
        
        self.df = pd.read_csv(filepath)
        self.pxs = self.df["px"].unique()
        self.pys = self.df["py"].unique()
        self.scs = self.df["sc_x"].unique()


    def _Get_Method_Durations_All(self, method):
        """
        NOTE: Square data only currently.

        Given a method (numpy, f2py77, f2py90), the mean duration and standard deviation
        of the duration are calculated for each combination of (px, py, sc_x).
        """
        df_method = self.df[self.df["method"] == method]
        df_method.drop("sc_y", axis=1, inplace=True)

        df_method = df_method.groupby(["px", "py", "sc_x"])
        return df_method.agg({"duration": "mean"}), df_method.agg({"duration": "std"})


    def Get_Numpy_Data(self):
        """
        Gets the Numpy durations for each px, py, sc.
        """
        self.numpy_data_mean, self.numpy_data_std = self._Get_Method_Durations_All("numpy")


    def Plot_Method_Data(self, method):
        n_px, n_py = len(self.pxs), len(self.pys)
        n_p = n_px * n_py

        # since data has been grouped & aggregated, only the 'duration' column remains;
        # so, split it evenly for the total number of processors
        method_data_mean, method_data_std = self._Get_Method_Durations_All(method)
        means = np.array_split(np.array(method_data_mean["duration"]), n_p)
        stds  = np.array_split(np.array(method_data_std["duration"]),  n_p)
        
        ig, ax = plt.subplots()
        ax.set_xscale("log", basex=2)

        for px in self.pxs:
            for py in self.pys:
                i = (px - 1) * n_px + (py - 1)
                ax.errorbar(self.scs, means[i], yerr=stds[i], label="$p_x=%d$, $p_y=%d$" % (px, py))

        plt.title("Total duration (%s) vs. grid axis length ($N$)" % method)
        plt.xlabel("$N$")
        plt.ylabel("Duration (s)")
        ax.legend(loc="best")


    def _Get_Processor_Durations(self, px, py):
        """
        NOTE: Square data only currently.

        Given the number of processors for x and y (px, py), the mean duration and
        standard deviation of the duration are calculated for each method and each sc_x.
        """
        df_p = self.df[ (self.df["px"] == px) & (self.df["py"] == py) ]
        df_p.drop("sc_y", axis=1, inplace=True)

        df_p = df_p.groupby(["method", "sc_x"])
        return df_p.agg({"duration": "mean"}), df_p.agg({"duration": "std"})


    def Plot_Processor_Durations(self, px, py):
        # since data has been grouped & aggregated, only the 'duration' column remains;
        # split it evenly (3-way), one for each method
        proc_data_mean, proc_data_std = self._Get_Processor_Durations(px, py)
        means = np.array_split(np.array(proc_data_mean["duration"]), 3)
        stds  = np.array_split(np.array(proc_data_std["duration"]),  3)
        
        ig, ax = plt.subplots()
        ax.set_xscale("log", basex=2)
        ax.errorbar(self.scs, means[0], yerr=stds[0], label="f2py77")
        ax.errorbar(self.scs, means[1], yerr=stds[1], label="f2py90")
        ax.errorbar(self.scs, means[2], yerr=stds[2], label="numpy")

        plt.title( "Total duration of specific processor configuration vs. grid axis length ($N$)\n" +
                    "($p_x=%d$, $p_y=%d$)" % (px, py) )
        plt.xlabel("$N$")
        plt.ylabel("Duration (s)")
        ax.legend(loc="best")


a = Analyzer()
a.Plot_Processor_Durations(px=1, py=1)
plt.show()
plt.clf()

a.Plot_Method_Data(method="numpy")
plt.show()
plt.clf()
