use std::{fs::File, io::Write};

pub trait ExportToJs {
    fn export_to_js(&self, js_file: &mut File, variable_name: &str) {
        self.export_to_js_checked(js_file, variable_name)
            .expect("could write to file");
    }
    fn export_to_js_checked(&self, js_file: &mut File, variable_name: &str) -> std::io::Result<()>;
}

#[derive(Debug, Clone)]
pub struct ExportedVariables {
    name: &'static str,
    list: Vec<String>,
}

impl ExportedVariables {
    pub fn new(name: &'static str) -> ExportedVariables {
        ExportedVariables { name, list: vec![] }
    }

    pub fn push(&mut self, var: impl Into<String>) -> &Self {
        self.list.push(var.into());
        self
    }

    pub fn export(&self, file: &mut File) {
        self.export_to_js(file, self.name);
    }
}

// ExportToJs Implentations:

impl ExportToJs for Vec<f64> {
    fn export_to_js_checked(&self, js_file: &mut File, variable_name: &str) -> std::io::Result<()> {
        writeln!(js_file, "let {variable_name} = '{:?}';", self)
    }
}

impl ExportToJs for Vec<[f64; 1]> {
    fn export_to_js_checked(&self, js_file: &mut File, variable_name: &str) -> std::io::Result<()> {
        self.iter()
            .map(|x| x[0])
            .collect::<Vec<f64>>()
            .export_to_js_checked(js_file, variable_name)
    }
}

//impl<const IN: usize, const OUT: usize> ExportToJs for TestsResult<IN, OUT> {
impl ExportToJs for TestsResult<1, 1> {
    fn export_to_js_checked(&self, js_file: &mut File, variable_name: &str) -> std::io::Result<()> {
        writeln!(
            js_file,
            "let {variable_name} = {{ gen: {}, error: '{}', outputs: {:?} }};",
            self.generation,
            self.error,
            self.outputs.iter().flatten().collect::<Vec<&f64>>()
        )
    }
}

impl ExportToJs for ExportedVariables {
    fn export_to_js_checked(&self, js_file: &mut File, variable_name: &str) -> std::io::Result<()> {
        // var allows multiple exports with the same variable_name
        writeln!(
            js_file,
            "var {variable_name} = [{}];",
            self.list.iter().map(ToString::to_string).join(",")
        )
    }
}

impl<const IN: usize, const OUT: usize> ExportToJs for TrainingsResult<'_, IN, OUT> {
    fn export_to_js_checked(&self, js_file: &mut File, variable_name: &str) -> std::io::Result<()> {
        writeln!(
            js_file,
            "let {variable_name} = [{{ gen: {}, output: {:?} }}];",
            self.generation, self.output,
        )
    }
}

// old
#[macro_export]
macro_rules! export_to_js {
    ( $js_file:expr => $( $var:ident = $val:expr ),* ) => {{
        use std::io::prelude::Write;
        let file: &mut ::std::fs::File = $js_file;
        $(
            writeln!(file, "let {} = {};", stringify!($var), $val).expect("could write to file");
        )*
    }};
    ( $js_file:expr => $( $var:ident ),* ) => {
        export_to_js!($js_file => $( $var = format!("'{:?}'", $var)),*)
    };
}
pub use export_to_js;
use itertools::Itertools;

use crate::results::{TestsResult, TrainingsResult};
