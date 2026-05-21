use num_complex::Complex;
use num_traits::Float;
use plotly::{Plot, Scatter, Layout};
use plotly::common::Mode;
use plotly::layout::{Axis, GridPattern, LayoutGrid};

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

/// Plot two spectra side-by-side in a single window.
///
/// Renders the two `(freqs, magnitude_db, subtitle)` traces as a 1-row × 2-column
/// subplot grid so different sample rates (or pre/post-resampling spectra) can be
/// compared visually. Each panel uses its own x-axis range; both share the same
/// "Magnitude (dB)" y-axis title.
///
/// # Arguments
/// * `left` - `(freqs, spectrum_db, subtitle)` for the left panel.
/// * `right` - `(freqs, spectrum_db, subtitle)` for the right panel.
/// * `title` - Overall figure title.
pub fn plot_spectrum_pair<T: Float>(
    left: (&Vec<T>, &Vec<T>, &str),
    right: (&Vec<T>, &Vec<T>, &str),
    title: &str,
) {
    let mut plot = Plot::new();
    plot.add_trace(spectrum_subplot_trace(left.0, left.1, left.2, "x1", "y1"));
    plot.add_trace(spectrum_subplot_trace(right.0, right.1, right.2, "x2", "y2"));
    plot.set_layout(side_by_side_spectrum_layout(title));
    plot.show();
}

/// Builds one frequency-vs-dB trace attached to the given subplot axes.
fn spectrum_subplot_trace<T: Float>(
    freqs: &[T],
    spectrum: &[T],
    name: &str,
    x_axis: &str,
    y_axis: &str,
) -> Box<Scatter<f64, f64>> {
    let freqs_f64: Vec<f64> = freqs.iter().map(|f| f.to_f64().unwrap()).collect();
    let spectrum_f64: Vec<f64> = spectrum.iter().map(|s| s.to_f64().unwrap()).collect();
    Scatter::new(freqs_f64, spectrum_f64)
        .mode(Mode::Lines)
        .name(name)
        .x_axis(x_axis)
        .y_axis(y_axis)
}

/// Returns the layout for a 1-row × 2-column side-by-side spectrum figure.
fn side_by_side_spectrum_layout(title: &str) -> Layout {
    Layout::new()
        .title(title)
        .grid(
            LayoutGrid::new()
                .rows(1)
                .columns(2)
                .pattern(GridPattern::Independent),
        )
        .x_axis(Axis::new().title("Frequency (Hz)").domain(&[0.0, 0.45]))
        .y_axis(Axis::new().title("Magnitude (dB)").domain(&[0.0, 1.0]))
        .x_axis2(Axis::new().title("Frequency (Hz)").domain(&[0.55, 1.0]))
        .y_axis2(Axis::new().title("Magnitude (dB)").domain(&[0.0, 1.0]))
        .auto_size(true)
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
