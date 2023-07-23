use crate::results::{TestsResult, TrainingsResult};
use itertools::Itertools;
use std::{fmt::Display, fs::File, io::Write};

pub enum JsVariableType {
    Var,
    Let,
    Const,
}

impl JsVariableType {
    pub const fn as_str(&self) -> &'static str {
        match self {
            JsVariableType::Var => "var",
            JsVariableType::Let => "let",
            JsVariableType::Const => "const",
        }
    }
}

pub trait ExportToJs {
    const JS_VARIABLE_TYPE: JsVariableType = JsVariableType::Let;
    /// defines how the value of the exported js variable looks like
    fn get_js_value(&self) -> String;

    fn export_to_js_checked(
        &self,
        js_file: &mut std::fs::File,
        variable_name: impl Display,
    ) -> std::io::Result<()> {
        writeln!(
            js_file,
            "{} {variable_name} = {};",
            Self::JS_VARIABLE_TYPE.as_str(),
            self.get_js_value()
        )
    }

    /// like `Self::export_to_js` but panics if file write fails
    fn export_to_js(&self, js_file: &mut File, variable_name: impl Display) {
        self.export_to_js_checked(js_file, variable_name).expect("could write to file");
    }
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
    fn get_js_value(&self) -> String {
        format!("'{:?}'", self)
    }
}

impl ExportToJs for Vec<[f64; 1]> {
    fn get_js_value(&self) -> String {
        self.iter().map(|x| x[0]).collect::<Vec<f64>>().get_js_value()
    }
}

impl ExportToJs for TestsResult<1> {
    fn get_js_value(&self) -> String {
        format!(
            "{{ error: '{}', outputs: {:?} }}",
            self.error,
            self.outputs.iter().map(|a| a.0).flatten().collect::<Vec<f64>>()
        )
    }
}

impl ExportToJs for ExportedVariables {
    // allow duplicate variable definitions
    const JS_VARIABLE_TYPE: JsVariableType = JsVariableType::Var;

    fn get_js_value(&self) -> String {
        format!("[{}]", self.list.iter().map(ToString::to_string).join(","))
    }
}

impl<const IN: usize, const OUT: usize> ExportToJs for TrainingsResult<'_, IN, OUT> {
    fn get_js_value(&self) -> String {
        format!("[{{ gen: {}, output: {:?} }}]", self.generation, self.output,)
    }
}
