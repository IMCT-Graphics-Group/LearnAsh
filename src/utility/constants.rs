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

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;
