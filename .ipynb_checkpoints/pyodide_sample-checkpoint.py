import numpy as np
from IPython.display import  HTML

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve
)

class RocPrCmDisplay():

    def __init__(self):
        pass
    # def __init__(self,estimator,x_test,y_test,pos_label=None):
    #     # Keep score and limits
    #     self.y_score = estimator.decision_function(x_test)
    #     self.min_score = np.min(self.y_score)
    #     self.max_score = np.max(self.y_score)
        
    #     # Set default pos label
    #     if pos_label is None:
    #         pos_label = 1
        
    #     # Create ROC curve
    #     self.fpr,self.tpr,self.roc_th = roc_curve(
    #         y_test,self.y_score,pos_label=pos_label
    #     )
        
    #     # Create PR curve
    #     self.prec,self.recall,self.pr_th = precision_recall_curve(
    #         y_test,self.y_score,pos_label=pos_label
    #     )

    #     # Convert Y test to boolean array
    #     self.y_test = y_test == pos_label

    def html(self):
        return HTML(''' 
        <!doctype html>
        <meta charset="utf-8">
        <html>                    
        <head>
            <title>Demo</title>
            <script src="https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js"></script>
        </head>
        <body>
            <script type="text/javascript">
            async function main() {
            let pyodide = await loadPyodide();
            console.log(
                pyodide.runPython(`
                    import sys
                    sys.version
                `)
            );
            await pyodide.loadPackage("matplotlib");
            pyodide.runPython(`
            import matplotlib.pyplot as plt
            import numpy as np

            from matplotlib.widgets import Button, Slider

            def f(t, amplitude, frequency):
                return amplitude * np.sin(2 * np.pi * frequency * t)

            t = np.linspace(0, 1, 1000)

            init_amplitude = 5
            init_frequency = 3

            fig, ax = plt.subplots()
            line, = ax.plot(t, f(t, init_amplitude, init_frequency), lw=2)
            ax.set_xlabel('Time [s]')

            fig.subplots_adjust(left=0.25, bottom=0.25)

            axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
            freq_slider = Slider(
                ax=axfreq,
                label='Frequency [Hz]',
                valmin=0.1,
                valmax=30,
                valinit=init_frequency,
            )

            axamp = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
            amp_slider = Slider(
                ax=axamp,
                label="Amplitude",
                valmin=0,
                valmax=10,
                valinit=init_amplitude,
                orientation="vertical"
            )

            def update(val):
                line.set_ydata(f(t, amp_slider.val, freq_slider.val))
                fig.canvas.draw_idle()

            freq_slider.on_changed(update)
            amp_slider.on_changed(update)

            resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
            button = Button(resetax, 'Reset', hovercolor='0.975')

            def reset(event):
                freq_slider.reset()
                amp_slider.reset()
            button.on_clicked(reset)

            plt.show()
            `);
            }
            main();
        </script>
        </body>
        <html>
        ''')
