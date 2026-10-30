// No console window in release builds on Windows.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--export-bindings") {
        let path = args.get(2).map(String::as_str).unwrap_or("web/src/engine/bindings.ts");
        if let Err(e) = rustyfem_app_lib::export_bindings(path) {
            eprintln!("{e}");
            std::process::exit(1);
        }
        return;
    }
    if args.get(1).map(String::as_str) == Some("--mcp") {
        rustyfem_app_lib::mcp::stdio_bridge();
        return;
    }
    if args.get(1).map(String::as_str) == Some("--selftest") {
        rustyfem_app_lib::run_selftest(args.get(2).map(std::path::PathBuf::from).unwrap_or_else(|| "selftest".into()));
        return;
    }
    rustyfem_app_lib::run()
}
