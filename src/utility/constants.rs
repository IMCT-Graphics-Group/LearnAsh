use std::os::raw::c_char;

use crate::utility::debug::ValidationInfo;
use crate::utility::structures::*;

use ash::vk;
use winit::event::VirtualKeyCode;

// Constants
pub const WINDOW_TITLE: &'static str = "测试窗口";
pub const WINDOW_WIDTH: u32 = 800;
pub const WINDOW_HEIGHT: u32 = 600;
pub const WINDOW_KEYCODE_EXIT: VirtualKeyCode = VirtualKeyCode::Escape;

pub const VALIDATION: ValidationInfo = ValidationInfo {
    is_enable: true,
    required_validation_layers: ["VK_LAYER_KHRONOS_validation"],
};
pub const APPLICATION_VERSION: u32 = vk::make_api_version(0, 1, 0, 0);
pub const API_VERSION: u32 = vk::make_api_version(0, 1, 0, 0);
pub const ENGINE_VERSION: u32 = vk::make_api_version(0, 1, 0, 0);
pub const DEVICE_EXTENSIONS: DeviceExtension = DeviceExtension {
    names: ["VK_KHR_swapchain"],
};

impl DeviceExtension {
    pub fn get_extensions_raw_names(&self) -> [*const c_char; 1] {
        [ash::extensions::khr::Swapchain::name().as_ptr()]
    }
}

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub const VERTICES_DATA: [Vertex; 4] = [
    Vertex {
        pos: [-0.5, -0.5],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        pos: [0.5, -0.5],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        pos: [0.5, 0.5],
        color: [0.0, 0.0, 1.0],
    },
    Vertex {
        pos: [-0.5, 0.5],
        color: [1.0, 1.0, 1.0],
    },
];

pub const INDICES_DATA: [u32; 6] = [0, 1, 2, 2, 3, 0];
