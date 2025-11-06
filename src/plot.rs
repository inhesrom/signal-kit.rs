use num_complex::Complex;
use num_traits::Float;
use plotly::{Plot, Scatter, Layout};
use plotly::common::Mode;
use plotly::layout::Axis;

/// Plot a spectrum or frequency response
///
/// This function creates a plot showing frequency vs magnitude (typically in dB).
///
/// # Arguments
/// * `freqs` - Frequency values for x-axis
/// * `spectrum` - Magnitude values for y-axis (typically in dB)
/// * `title` - Title for the plot
pub fn plot_spectrum<T: Float>(freqs: &Vec<T>, spectrum: &Vec<T>, title: &str) {
    // Convert to f64 for plotting
    let freqs_f64: Vec<f64> = freqs.iter().map(|f| f.to_f64().unwrap()).collect();
    let spectrum_f64: Vec<f64> = spectrum.iter().map(|s| s.to_f64().unwrap()).collect();

    let mut plot = Plot::new();
    let trace = Scatter::new(freqs_f64, spectrum_f64);
    plot.add_trace(trace);

    let layout = Layout::new()
        .title(title)
        .x_axis(Axis::new().title("Frequency (Hz)"))
        .y_axis(Axis::new().title("Magnitude (dB)"))
        .auto_size(true);
    plot.set_layout(layout);

    plot.show();
}

/// Plot a constellation diagram from a vector of complex symbols
///
/// This function creates an interactive plot showing the I/Q constellation
/// with equal-sized axes for proper visualization.
///
/// # Arguments
/// * `symbols` - Slice of complex symbols to plot
/// * `title` - Title for the plot
///
/// # Example
/// ```
/// use num_complex::Complex;
/// use signal_kit::plot::plot_constellation;
///
/// let symbols = vec![
///     Complex::new(1.0, 1.0),
///     Complex::new(-1.0, 1.0),
///     Complex::new(-1.0, -1.0),
///     Complex::new(1.0, -1.0),
/// ];
/// plot_constellation(&symbols, "QPSK Constellation");
/// ```
pub fn plot_constellation<T: Float + std::fmt::Display>(symbols: &[Complex<T>], title: &str) {
    // Extract I and Q values
    let mut i_vals = Vec::new();
    let mut q_vals = Vec::new();

    for symbol in symbols {
        // Convert to f64 for plotting
        i_vals.push(symbol.re.to_f64().unwrap());
        q_vals.push(symbol.im.to_f64().unwrap());
    }

    // Create scatter plot for constellation points
    let trace = Scatter::new(i_vals.clone(), q_vals.clone())
        .mode(Mode::Markers)
        .name("Constellation");

    let mut plot = Plot::new();
    plot.add_trace(trace);

    // Add axis reference lines at 0
    let x_axis_line = Scatter::new(vec![-1.5, 1.5], vec![0.0, 0.0])
        .mode(Mode::Lines)
        .name("I-axis")
        .show_legend(false);
    let y_axis_line = Scatter::new(vec![0.0, 0.0], vec![-1.5, 1.5])
        .mode(Mode::Lines)
        .name("Q-axis")
        .show_legend(false);

    plot.add_trace(x_axis_line);
    plot.add_trace(y_axis_line);

    // Set layout with equal aspect ratio using scaleanchor
    let layout = Layout::new()
        .title(title)
        .x_axis(Axis::new().title("In-Phase (I)"))
        .y_axis(
            Axis::new()
                .title("Quadrature (Q)")
                .scale_anchor("x")
                .constrain(plotly::layout::AxisConstrain::Domain)
        )
        .auto_size(true);
    plot.set_layout(layout);

    // Show the plot
    plot.show();
}
