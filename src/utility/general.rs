use crate::{utility, utility::constants::*, utility::platforms, utility::structures::*};

use std::{
    collections::HashSet,
    ffi::{c_char, c_void, CString},
    path::Path,
    ptr,
};

use ash::vk;

use super::debug::ValidationInfo;

pub fn create_instance(
    entry: &ash::Entry,
    window_title: &str,
    validation: &ValidationInfo,
) -> ash::Instance {
    if validation.is_enable
        && utility::debug::check_validation_layer_support(
            entry,
            &validation.required_validation_layers.to_vec(),
        ) == false
    {
        panic!("Validation layers requested, but not available!");
    }

    let app_name = CString::new(window_title).unwrap();
    let engine_name = CString::new("Vulkan Engine").unwrap();
    let app_info = vk::ApplicationInfo {
        s_type: vk::StructureType::APPLICATION_INFO,
        p_next: ptr::null(),
        p_application_name: app_name.as_ptr(),
        application_version: APPLICATION_VERSION,
        p_engine_name: engine_name.as_ptr(),
        engine_version: ENGINE_VERSION,
        api_version: API_VERSION,
    };

    let debug_utils_create_info = utility::debug::populate_debug_messenger_create_info();

    let extension_names = utility::platforms::required_extension_names();

    let required_validation_layer_raw_names: Vec<CString> = validation
        .required_validation_layers
        .iter()
        .map(|layer_name| CString::new(*layer_name).unwrap())
        .collect();
    let layer_names: Vec<*const i8> = required_validation_layer_raw_names
        .iter()
        .map(|layer_name| layer_name.as_ptr())
        .collect();

    let create_info = vk::InstanceCreateInfo {
        s_type: vk::StructureType::INSTANCE_CREATE_INFO,
        p_next: if validation.is_enable {
            &debug_utils_create_info as *const vk::DebugUtilsMessengerCreateInfoEXT as *const c_void
        } else {
            ptr::null()
        },
        flags: vk::InstanceCreateFlags::empty(),
        p_application_info: &app_info,
        pp_enabled_layer_names: if validation.is_enable {
            layer_names.as_ptr()
        } else {
            ptr::null()
        },
        enabled_layer_count: if validation.is_enable {
            layer_names.len()
        } else {
            0
        } as u32,
        pp_enabled_extension_names: extension_names.as_ptr(),
        enabled_extension_count: extension_names.len() as u32,
    };

    let instance: ash::Instance = unsafe {
        entry
            .create_instance(&create_info, None)
            .expect("Failed to create instance!")
    };

    instance
}

pub fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &winit::window::Window,
    screen_width: u32,
    screen_height: u32,
) -> SurfaceStuff {
    let surface = unsafe {
        platforms::create_surface(entry, instance, window).expect("Failed to create surface.")
    };
    let surface_loader = ash::extensions::khr::Surface::new(entry, instance);

    SurfaceStuff {
        surface_loader,
        surface,
        screen_width,
        screen_height,
    }
}

pub fn pick_physcial_device(
    instance: &ash::Instance,
    surface_stuff: &SurfaceStuff,
    required_device_extensions: &DeviceExtension,
) -> vk::PhysicalDevice {
    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .expect("Failed to enumerate Physical Devices!")
    };

    let result = physical_devices.iter().find(|physical_device| {
        let is_suitable = is_physical_device_suitable(
            instance,
            **physical_device,
            surface_stuff,
            required_device_extensions,
        );

        is_suitable
    });

    match result {
        Some(physical_device) => *physical_device,
        None => panic!("Failed to find a suitable GPU!"),
    }
}

fn is_physical_device_suitable(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface_stuff: &SurfaceStuff,
    required_device_extensions: &DeviceExtension,
) -> bool {
    let device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
    println!(
        "Current Device: {}",
        utility::tools::vk_to_string(&device_properties.device_name)
    );

    let device_features = unsafe { instance.get_physical_device_features(physical_device) };

    let indices = find_queue_family(instance, physical_device, surface_stuff);

    let is_queue_family_supported = indices.is_complete();
    let is_device_extension_supported =
        check_device_extension_support(instance, physical_device, required_device_extensions);

    let is_swapchain_supported = if is_device_extension_supported {
        let swapchain_support = query_swapchain_support(physical_device, surface_stuff);
        !swapchain_support.formats.is_empty() && !swapchain_support.present_modes.is_empty()
    } else {
        false
    };
    let is_support_sampler_anisotropy = device_features.sampler_anisotropy == 1;

    return is_queue_family_supported
        && is_device_extension_supported
        && is_swapchain_supported
        && is_support_sampler_anisotropy;
}

pub fn create_logical_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    validation: &ValidationInfo,
    device_extension: &DeviceExtension,
    surface_stuff: &SurfaceStuff,
) -> (ash::Device, QueueFamilyIndices) {
    let indices = find_queue_family(instance, physical_device, surface_stuff);

    let mut unique_queue_families = HashSet::new();
    unique_queue_families.insert(indices.graphics_family.unwrap());
    unique_queue_families.insert(indices.present_family.unwrap());

    let queue_priorities = [1.0_f32];
    let mut queue_create_infos = vec![];
    for &queue_family in unique_queue_families.iter() {
        let queue_create_info = vk::DeviceQueueCreateInfo {
            s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DeviceQueueCreateFlags::empty(),
            queue_family_index: queue_family,
            p_queue_priorities: queue_priorities.as_ptr(),
            queue_count: queue_priorities.len() as u32,
        };
        queue_create_infos.push(queue_create_info);
    }

    let physical_device_features = vk::PhysicalDeviceFeatures {
        sampler_anisotropy: vk::TRUE,
        ..Default::default()
    };

    let required_validation_layer_raw_names: Vec<CString> = validation
        .required_validation_layers
        .iter()
        .map(|layer_name| CString::new(*layer_name).unwrap())
        .collect();
    let enable_layer_names: Vec<*const c_char> = required_validation_layer_raw_names
        .iter()
        .map(|layer_name| layer_name.as_ptr())
        .collect();

    let enable_extension_names = device_extension.get_extensions_raw_names();

    let device_create_info = vk::DeviceCreateInfo {
        s_type: vk::StructureType::DEVICE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::DeviceCreateFlags::empty(),
        queue_create_info_count: queue_create_infos.len() as u32,
        p_queue_create_infos: queue_create_infos.as_ptr(),
        enabled_layer_count: if validation.is_enable {
            enable_layer_names.len()
        } else {
            0
        } as u32,
        pp_enabled_layer_names: if validation.is_enable {
            enable_layer_names.as_ptr()
        } else {
            ptr::null()
        },
        enabled_extension_count: enable_extension_names.len() as u32,
        pp_enabled_extension_names: enable_extension_names.as_ptr(),
        p_enabled_features: &physical_device_features,
    };

    let device: ash::Device = unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .expect("Failed to create logical Device!")
    };

    (device, indices)
}

fn find_queue_family(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface_stuff: &SurfaceStuff,
) -> QueueFamilyIndices {
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

    let mut queue_family_indices = QueueFamilyIndices::new();

    let mut index: u32 = 0;
    for queue_family in queue_families.iter() {
        if queue_family.queue_count > 0
            && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        {
            queue_family_indices.graphics_family = Some(index);
        }

        let is_present_support = unsafe {
            surface_stuff
                .surface_loader
                .get_physical_device_surface_support(physical_device, index, surface_stuff.surface)
                .unwrap()
        };

        if queue_family.queue_count > 0 && is_present_support {
            queue_family_indices.present_family = Some(index);
        }

        if queue_family_indices.is_complete() {
            break;
        }

        index += 1;
    }

    queue_family_indices
}

fn check_device_extension_support(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    device_extension: &DeviceExtension,
) -> bool {
    let available_extensions = unsafe {
        instance
            .enumerate_device_extension_properties(physical_device)
            .expect("Failed to get device extension properties.")
    };

    let mut available_extension_names = vec![];

    for extension in available_extensions.iter() {
        let extension_name = utility::tools::vk_to_string(&extension.extension_name);
        available_extension_names.push(extension_name);
    }

    let mut required_extensions = HashSet::new();
    for extension in device_extension.names.iter() {
        required_extensions.insert(extension.to_string());
    }

    for extension_name in available_extension_names.iter() {
        required_extensions.remove(extension_name);
    }

    return required_extensions.is_empty();
}

fn query_swapchain_support(
    physical_device: vk::PhysicalDevice,
    surface_stuff: &SurfaceStuff,
) -> SwapChainSupportDetail {
    unsafe {
        let capabilities = surface_stuff
            .surface_loader
            .get_physical_device_surface_capabilities(physical_device, surface_stuff.surface)
            .expect("Failed to query for surface capabilities.");
        let formats = surface_stuff
            .surface_loader
            .get_physical_device_surface_formats(physical_device, surface_stuff.surface)
            .expect("Failed to query for surface formats.");
        let present_modes = surface_stuff
            .surface_loader
            .get_physical_device_surface_present_modes(physical_device, surface_stuff.surface)
            .expect("Failed to query for surface present mode.");

        SwapChainSupportDetail {
            capabilities,
            formats,
            present_modes,
        }
    }
}

pub fn create_swapchain(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    window: &winit::window::Window,
    surface_stuff: &SurfaceStuff,
    queue_family: &QueueFamilyIndices,
) -> SwapChainStuff {
    let swapchain_support = query_swapchain_support(physical_device, surface_stuff);

    let surface_format = choose_swapchain_format(&swapchain_support.formats);
    let present_mode = choose_swapchain_present_mode(&swapchain_support.present_modes);
    let extent = choose_swapchain_extent(&swapchain_support.capabilities, window);

    let image_count = swapchain_support.capabilities.min_image_count + 1;
    let image_count = if swapchain_support.capabilities.max_image_count > 0 {
        image_count.min(swapchain_support.capabilities.max_image_count)
    } else {
        image_count
    };

    let (image_sharing_mode, queue_family_index_count, queue_family_indices) =
        if queue_family.graphics_family != queue_family.present_family {
            (
                vk::SharingMode::CONCURRENT,
                2,
                vec![
                    queue_family.graphics_family.unwrap(),
                    queue_family.present_family.unwrap(),
                ],
            )
        } else {
            (vk::SharingMode::EXCLUSIVE, 0, vec![])
        };

    let swapchain_create_info = vk::SwapchainCreateInfoKHR {
        s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
        p_next: ptr::null(),
        flags: vk::SwapchainCreateFlagsKHR::empty(),
        surface: surface_stuff.surface,
        min_image_count: image_count,
        image_color_space: surface_format.color_space,
        image_format: surface_format.format,
        image_extent: extent,
        image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        image_sharing_mode,
        p_queue_family_indices: queue_family_indices.as_ptr(),
        queue_family_index_count,
        pre_transform: swapchain_support.capabilities.current_transform,
        composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
        present_mode,
        clipped: vk::TRUE,
        old_swapchain: vk::SwapchainKHR::null(),
        image_array_layers: 1,
    };

    let swapchain_loader = ash::extensions::khr::Swapchain::new(instance, device);
    let swapchain = unsafe {
        swapchain_loader
            .create_swapchain(&swapchain_create_info, None)
            .expect("Failed to create Swapchain!")
    };

    let swapchain_images = unsafe {
        swapchain_loader
            .get_swapchain_images(swapchain)
            .expect("Failed to get Swapcahin Images.")
    };

    SwapChainStuff {
        swapchain_loader,
        swapchain,
        swapchain_images,
        swapchain_format: surface_format.format,
        swapchain_extent: extent,
    }
}

fn choose_swapchain_format(available_formats: &Vec<vk::SurfaceFormatKHR>) -> vk::SurfaceFormatKHR {
    for available_format in available_formats {
        if available_format.format == vk::Format::B8G8R8A8_SRGB
            && available_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        {
            return available_format.clone();
        }
    }

    return available_formats.first().unwrap().clone();
}

fn choose_swapchain_present_mode(
    available_present_modes: &Vec<vk::PresentModeKHR>,
) -> vk::PresentModeKHR {
    for &available_present_mode in available_present_modes.iter() {
        if available_present_mode == vk::PresentModeKHR::MAILBOX {
            return available_present_mode;
        }
    }

    vk::PresentModeKHR::FIFO
}

fn choose_swapchain_extent(
    capabilities: &vk::SurfaceCapabilitiesKHR,
    window: &winit::window::Window,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        use num::clamp;

        let window_size = window.inner_size();
        vk::Extent2D {
            width: clamp(
                window_size.width as u32,
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: clamp(
                window_size.height as u32,
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    }
}

pub fn create_image_views(
    device: &ash::Device,
    surface_format: vk::Format,
    images: &Vec<vk::Image>,
) -> Vec<vk::ImageView> {
    let swapchain_imageviews: Vec<vk::ImageView> = images
        .iter()
        .map(|&image| {
            create_image_view(
                device,
                image,
                surface_format,
                vk::ImageAspectFlags::COLOR,
                1,
            )
        })
        .collect();

    swapchain_imageviews
}

pub fn create_image_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    aspect_flags: vk::ImageAspectFlags,
    mip_levels: u32,
) -> vk::ImageView {
    let imageview_create_info = vk::ImageViewCreateInfo {
        s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::ImageViewCreateFlags::empty(),
        view_type: vk::ImageViewType::TYPE_2D,
        format,
        components: vk::ComponentMapping {
            r: vk::ComponentSwizzle::IDENTITY,
            g: vk::ComponentSwizzle::IDENTITY,
            b: vk::ComponentSwizzle::IDENTITY,
            a: vk::ComponentSwizzle::IDENTITY,
        },
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: aspect_flags,
            base_mip_level: 0,
            level_count: mip_levels,
            base_array_layer: 0,
            layer_count: 1,
        },
        image,
    };

    unsafe {
        device
            .create_image_view(&imageview_create_info, None)
            .expect("Failed to create Image View!")
    }
}

pub fn create_graphics_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    swapchain_extent: vk::Extent2D,
    ubo_set_layout: vk::DescriptorSetLayout,
) -> (vk::Pipeline, vk::PipelineLayout) {
    let vert_shader_code = utility::tools::read_shader_code(Path::new("shaders/spv/vert.spv"));
    let frag_shader_code = utility::tools::read_shader_code(Path::new("shaders/spv/frag.spv"));

    let vert_shader_module = create_shader_module(device, vert_shader_code);
    let frag_shader_module = create_shader_module(device, frag_shader_code);

    let main_function_name = CString::new("main").unwrap();

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            module: vert_shader_module,
            p_name: main_function_name.as_ptr(),
            p_specialization_info: ptr::null(),
            stage: vk::ShaderStageFlags::VERTEX,
        },
        vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            module: frag_shader_module,
            p_name: main_function_name.as_ptr(),
            p_specialization_info: ptr::null(),
            stage: vk::ShaderStageFlags::FRAGMENT,
        },
    ];

    let binding_description = Vertex::get_binding_descriptions();
    let attribute_description = Vertex::get_attribute_descriptions();

    let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineVertexInputStateCreateFlags::empty(),
        vertex_attribute_description_count: attribute_description.len() as u32,
        p_vertex_attribute_descriptions: attribute_description.as_ptr(),
        vertex_binding_description_count: binding_description.len() as u32,
        p_vertex_binding_descriptions: binding_description.as_ptr(),
    };
    let vertex_input_assembly_state_create_info = vk::PipelineInputAssemblyStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        flags: vk::PipelineInputAssemblyStateCreateFlags::empty(),
        p_next: ptr::null(),
        primitive_restart_enable: vk::FALSE,
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
    };

    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: swapchain_extent.width as f32,
        height: swapchain_extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }];

    let scissors = [vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: swapchain_extent,
    }];

    // let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

    // let dynamic_state_info = vk::PipelineDynamicStateCreateInfo {
    //     s_type: vk::StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO,
    //     p_next: ptr::null(),
    //     flags: vk::PipelineDynamicStateCreateFlags::empty(),
    //     dynamic_state_count: dynamic_state.len() as u32,
    //     p_dynamic_states: dynamic_state.as_ptr(),
    // };

    let viewport_state_create_info = vk::PipelineViewportStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineViewportStateCreateFlags::empty(),
        scissor_count: scissors.len() as u32,
        p_scissors: scissors.as_ptr(),
        viewport_count: viewports.len() as u32,
        p_viewports: viewports.as_ptr(),
    };

    let rasterization_state_create_info = vk::PipelineRasterizationStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineRasterizationStateCreateFlags::empty(),
        depth_clamp_enable: vk::FALSE,
        cull_mode: vk::CullModeFlags::BACK,
        front_face: vk::FrontFace::CLOCKWISE,
        line_width: 1.0,
        polygon_mode: vk::PolygonMode::FILL,
        rasterizer_discard_enable: vk::FALSE,
        depth_bias_clamp: 0.0,
        depth_bias_constant_factor: 0.0,
        depth_bias_enable: vk::FALSE,
        depth_bias_slope_factor: 0.0,
    };
    let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineMultisampleStateCreateFlags::empty(),
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        sample_shading_enable: vk::FALSE,
        min_sample_shading: 0.0,
        p_sample_mask: ptr::null(),
        alpha_to_one_enable: vk::FALSE,
        alpha_to_coverage_enable: vk::FALSE,
    };

    let stencil_state = vk::StencilOpState {
        fail_op: vk::StencilOp::KEEP,
        pass_op: vk::StencilOp::KEEP,
        depth_fail_op: vk::StencilOp::KEEP,
        compare_op: vk::CompareOp::ALWAYS,
        compare_mask: 0,
        write_mask: 0,
        reference: 0,
    };

    let depth_state_create_info = vk::PipelineDepthStencilStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineDepthStencilStateCreateFlags::empty(),
        depth_test_enable: vk::FALSE,
        depth_write_enable: vk::FALSE,
        depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
        depth_bounds_test_enable: vk::FALSE,
        stencil_test_enable: vk::FALSE,
        front: stencil_state,
        back: stencil_state,
        max_depth_bounds: 1.0,
        min_depth_bounds: 0.0,
    };

    let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
        blend_enable: vk::FALSE,
        color_write_mask: vk::ColorComponentFlags::RGBA,
        src_color_blend_factor: vk::BlendFactor::ONE,
        dst_color_blend_factor: vk::BlendFactor::ZERO,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
    }];

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineColorBlendStateCreateFlags::empty(),
        logic_op_enable: vk::FALSE,
        logic_op: vk::LogicOp::COPY,
        attachment_count: color_blend_attachment_states.len() as u32,
        p_attachments: color_blend_attachment_states.as_ptr(),
        blend_constants: [0.0, 0.0, 0.0, 0.0],
    };

    let set_layouts = [ubo_set_layout];

    let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
        s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineLayoutCreateFlags::empty(),
        set_layout_count: set_layouts.len() as u32,
        p_set_layouts: set_layouts.as_ptr(),
        push_constant_range_count: 0,
        p_push_constant_ranges: ptr::null(),
    };

    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&pipeline_layout_create_info, None)
            .expect("Failed to create pipeline layout!")
    };

    let graphics_pipeline_create_infos = [vk::GraphicsPipelineCreateInfo {
        s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineCreateFlags::empty(),
        stage_count: shader_stages.len() as u32,
        p_stages: shader_stages.as_ptr(),
        p_vertex_input_state: &vertex_input_state_create_info,
        p_input_assembly_state: &vertex_input_assembly_state_create_info,
        p_tessellation_state: ptr::null(),
        p_viewport_state: &viewport_state_create_info,
        p_rasterization_state: &rasterization_state_create_info,
        p_multisample_state: &multisample_state_create_info,
        p_depth_stencil_state: &depth_state_create_info,
        p_color_blend_state: &color_blend_state,
        p_dynamic_state: ptr::null(),
        layout: pipeline_layout,
        render_pass,
        subpass: 0,
        base_pipeline_handle: vk::Pipeline::null(),
        base_pipeline_index: -1,
    }];

    let graphics_pipelines = unsafe {
        device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &graphics_pipeline_create_infos,
                None,
            )
            .expect("Failed to create Graphics Pipeline!")
    };

    unsafe {
        device.destroy_shader_module(vert_shader_module, None);
        device.destroy_shader_module(frag_shader_module, None);
    }

    (graphics_pipelines[0], pipeline_layout)
}

fn create_shader_module(device: &ash::Device, code: Vec<u8>) -> vk::ShaderModule {
    let shader_module_create_info = vk::ShaderModuleCreateInfo {
        s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::ShaderModuleCreateFlags::empty(),
        code_size: code.len(),
        p_code: code.as_ptr() as *const u32,
    };

    unsafe {
        device
            .create_shader_module(&shader_module_create_info, None)
            .expect("Failed to create Shader Module!")
    }
}

pub fn create_render_pass(device: &ash::Device, surface_format: vk::Format) -> vk::RenderPass {
    let color_attachment = vk::AttachmentDescription {
        flags: vk::AttachmentDescriptionFlags::empty(),
        format: surface_format,
        samples: vk::SampleCountFlags::TYPE_1,
        load_op: vk::AttachmentLoadOp::CLEAR,
        store_op: vk::AttachmentStoreOp::STORE,
        stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout: vk::ImageLayout::UNDEFINED,
        final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
    };

    let color_attachment_ref = vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    };

    let subpasses = [vk::SubpassDescription {
        flags: vk::SubpassDescriptionFlags::empty(),
        pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
        input_attachment_count: 0,
        p_input_attachments: ptr::null(),
        color_attachment_count: 1,
        p_color_attachments: &color_attachment_ref,
        p_resolve_attachments: ptr::null(),
        p_depth_stencil_attachment: ptr::null(),
        preserve_attachment_count: 0,
        p_preserve_attachments: ptr::null(),
    }];

    let render_pass_attachments = [color_attachment];

    let subpass_dependencies = [vk::SubpassDependency {
        src_subpass: vk::SUBPASS_EXTERNAL,
        dst_subpass: 0,
        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        src_access_mask: vk::AccessFlags::empty(),
        dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dependency_flags: vk::DependencyFlags::empty(),
    }];

    let renderpass_create_info = vk::RenderPassCreateInfo {
        s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
        flags: vk::RenderPassCreateFlags::empty(),
        p_next: ptr::null(),
        attachment_count: render_pass_attachments.len() as u32,
        p_attachments: render_pass_attachments.as_ptr(),
        subpass_count: subpasses.len() as u32,
        p_subpasses: subpasses.as_ptr(),
        dependency_count: subpass_dependencies.len() as u32,
        p_dependencies: subpass_dependencies.as_ptr(),
    };

    unsafe {
        device
            .create_render_pass(&renderpass_create_info, None)
            .expect("Failed to create render pass!")
    }
}

pub fn create_descriptor_set_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
    let ubo_layout_bindings = [vk::DescriptorSetLayoutBinding {
        binding: 0,
        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::VERTEX,
        p_immutable_samplers: ptr::null(),
    }];

    let ubo_layout_create_info = vk::DescriptorSetLayoutCreateInfo {
        s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::DescriptorSetLayoutCreateFlags::empty(),
        binding_count: ubo_layout_bindings.len() as u32,
        p_bindings: ubo_layout_bindings.as_ptr(),
    };

    unsafe {
        device
            .create_descriptor_set_layout(&ubo_layout_create_info, None)
            .expect("Failed to create Descriptor Set Layout!")
    }
}

pub fn create_uniform_buffers(
    device: &ash::Device,
    device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    swapchain_image_count: usize,
) -> (Vec<vk::Buffer>, Vec<vk::DeviceMemory>) {
    let buffer_size = std::mem::size_of::<UniformBufferObject>();

    let mut uniform_buffers = vec![];
    let mut uniform_buffers_memory = vec![];

    for _ in 0..swapchain_image_count {
        let (uniform_buffer, uniform_buffer_memory) = create_buffer(
            device,
            buffer_size as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            device_memory_properties,
        );
        uniform_buffers.push(uniform_buffer);
        uniform_buffers_memory.push(uniform_buffer_memory)
    }

    (uniform_buffers, uniform_buffers_memory)
}

pub fn create_framebuffers(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    image_views: &Vec<vk::ImageView>,
    swapchain_extent: vk::Extent2D,
) -> Vec<vk::Framebuffer> {
    let mut framebuffers = vec![];

    for &image_view in image_views.iter() {
        let attachments = [image_view];

        let framebuffer_create_info = vk::FramebufferCreateInfo {
            s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::FramebufferCreateFlags::empty(),
            render_pass,
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            width: swapchain_extent.width,
            height: swapchain_extent.height,
            layers: 1,
        };

        let framebuffer = unsafe {
            device
                .create_framebuffer(&framebuffer_create_info, None)
                .expect("Failed to create Framebuffer!")
        };

        framebuffers.push(framebuffer);
    }

    framebuffers
}

pub fn create_command_pool(
    device: &ash::Device,
    queue_families: &QueueFamilyIndices,
) -> vk::CommandPool {
    let command_pool_create_info = vk::CommandPoolCreateInfo {
        s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::CommandPoolCreateFlags::empty(),
        queue_family_index: queue_families.graphics_family.unwrap(),
    };

    unsafe {
        device
            .create_command_pool(&command_pool_create_info, None)
            .expect("Failed to create Command Pool!")
    }
}

pub fn create_vertex_buffer(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    submit_queue: vk::Queue,
) -> (vk::Buffer, vk::DeviceMemory) {
    let buffer_size = std::mem::size_of_val(&VERTICES_DATA) as vk::DeviceSize;
    let device_memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        device,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        &device_memory_properties,
    );

    unsafe {
        let data_ptr = device
            .map_memory(
                staging_buffer_memory,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
            )
            .expect("Failed to Map Memory!") as *mut Vertex;

        data_ptr.copy_from_nonoverlapping(VERTICES_DATA.as_ptr(), VERTICES_DATA.len());

        device.unmap_memory(staging_buffer_memory);
    }

    let (vertex_buffer, vertex_buffer_memory) = create_buffer(
        device,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        &device_memory_properties,
    );

    copy_buffer(
        device,
        submit_queue,
        command_pool,
        staging_buffer,
        vertex_buffer,
        buffer_size,
    );

    unsafe {
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);
    }

    (vertex_buffer, vertex_buffer_memory)
}

pub fn create_index_buffer(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    submit_queue: vk::Queue,
) -> (vk::Buffer, vk::DeviceMemory) {
    let buffer_size = std::mem::size_of_val(&INDICES_DATA) as vk::DeviceSize;
    let device_memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let (staging_buffer, staging_buffer_memory) = create_buffer(
        device,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        &device_memory_properties,
    );

    unsafe {
        let data_ptr = device
            .map_memory(
                staging_buffer_memory,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
            )
            .expect("Failed to Map Memory!") as *mut u32;

        data_ptr.copy_from_nonoverlapping(INDICES_DATA.as_ptr(), INDICES_DATA.len());

        device.unmap_memory(staging_buffer_memory);
    }

    let (index_buffer, index_buffer_memory) = create_buffer(
        device,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        &device_memory_properties,
    );

    copy_buffer(
        device,
        submit_queue,
        command_pool,
        staging_buffer,
        index_buffer,
        buffer_size,
    );

    unsafe {
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);
    }

    (index_buffer, index_buffer_memory)
}

fn create_buffer(
    device: &ash::Device,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    required_memory_properties: vk::MemoryPropertyFlags,
    device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
) -> (vk::Buffer, vk::DeviceMemory) {
    let buffer_create_info = vk::BufferCreateInfo {
        s_type: vk::StructureType::BUFFER_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::BufferCreateFlags::empty(),
        size,
        usage,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        queue_family_index_count: 0,
        p_queue_family_indices: ptr::null(),
    };

    let buffer = unsafe {
        device
            .create_buffer(&buffer_create_info, None)
            .expect("Failed to create Vertex Buffer!")
    };

    let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory_type = find_memory_type(
        mem_requirements.memory_type_bits,
        required_memory_properties,
        device_memory_properties,
    );

    let allocate_info = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        p_next: ptr::null(),
        allocation_size: mem_requirements.size,
        memory_type_index: memory_type,
    };

    let buffer_memory = unsafe {
        device
            .allocate_memory(&allocate_info, None)
            .expect("Failed to allocate vertex buffer memory!")
    };

    unsafe {
        device
            .bind_buffer_memory(buffer, buffer_memory, 0)
            .expect("Failed to bind Buffer!");
    }

    (buffer, buffer_memory)
}

fn copy_buffer(
    device: &ash::Device,
    submit_queue: vk::Queue,
    command_pool: vk::CommandPool,
    src_buffer: vk::Buffer,
    dst_buffer: vk::Buffer,
    size: vk::DeviceSize,
) {
    let allocate_info = vk::CommandBufferAllocateInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        p_next: ptr::null(),
        command_buffer_count: 1,
        command_pool,
        level: vk::CommandBufferLevel::PRIMARY,
    };

    let command_buffers = unsafe {
        device
            .allocate_command_buffers(&allocate_info)
            .expect("Failed to allocate Command Buffer")
    };
    let command_buffer = command_buffers[0];

    let begin_info = vk::CommandBufferBeginInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
        p_next: ptr::null(),
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        p_inheritance_info: ptr::null(),
    };

    unsafe {
        device
            .begin_command_buffer(command_buffer, &begin_info)
            .expect("Failed to begin Command Buffer");

        let copy_regions = [vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size,
        }];

        device.cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, &copy_regions);

        device
            .end_command_buffer(command_buffer)
            .expect("Failed to end Command Buffer");
    }

    let submit_info = [vk::SubmitInfo {
        s_type: vk::StructureType::SUBMIT_INFO,
        p_next: ptr::null(),
        wait_semaphore_count: 0,
        p_wait_semaphores: ptr::null(),
        p_wait_dst_stage_mask: ptr::null(),
        command_buffer_count: 1,
        p_command_buffers: &command_buffer,
        signal_semaphore_count: 0,
        p_signal_semaphores: ptr::null(),
    }];

    unsafe {
        device
            .queue_submit(submit_queue, &submit_info, vk::Fence::null())
            .expect("Failed to Submit Queue!");
        device
            .queue_wait_idle(submit_queue)
            .expect("Failed to wait Queue idle!");

        device.free_command_buffers(command_pool, &command_buffers);
    }
}

fn find_memory_type(
    type_filter: u32,
    required_properties: vk::MemoryPropertyFlags,
    mem_properties: &vk::PhysicalDeviceMemoryProperties,
) -> u32 {
    for (i, memory_type) in mem_properties.memory_types.iter().enumerate() {
        if (type_filter & (1 << i)) > 0 && memory_type.property_flags.contains(required_properties)
        {
            return i as u32;
        }
    }

    panic!("Failed to find suitable memory type!")
}

pub fn create_command_buffers(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    graphics_pipeline: vk::Pipeline,
    framebuffers: &Vec<vk::Framebuffer>,
    render_pass: vk::RenderPass,
    surface_extent: vk::Extent2D,
    vertex_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
    pipeline_layout: vk::PipelineLayout,
    descriptor_sets: &Vec<vk::DescriptorSet>,
) -> Vec<vk::CommandBuffer> {
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        p_next: ptr::null(),
        command_buffer_count: framebuffers.len() as u32,
        command_pool,
        level: vk::CommandBufferLevel::PRIMARY,
    };

    let command_buffers = unsafe {
        device
            .allocate_command_buffers(&command_buffer_allocate_info)
            .expect("Failed to allocate Command Buffers!")
    };

    for (i, &command_buffer) in command_buffers.iter().enumerate() {
        let command_buffer_begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            p_inheritance_info: ptr::null(),
            flags: vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
        };

        unsafe {
            device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .expect("Failed to begin recording Command Buffer at beginning!");
        }

        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];

        let render_pass_begin_info = vk::RenderPassBeginInfo {
            s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
            p_next: ptr::null(),
            render_pass,
            framebuffer: framebuffers[i],
            render_area: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: surface_extent,
            },
            clear_value_count: clear_values.len() as u32,
            p_clear_values: clear_values.as_ptr(),
        };

        unsafe {
            device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                graphics_pipeline,
            );

            let vertex_buffers = [vertex_buffer];
            let offsets = [0_u64];
            let descriptor_sets_to_bind = [descriptor_sets[i]];

            device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
            device.cmd_bind_index_buffer(command_buffer, index_buffer, 0, vk::IndexType::UINT32);
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                0,
                &descriptor_sets_to_bind,
                &[],
            );

            device.cmd_draw_indexed(command_buffer, INDICES_DATA.len() as u32, 1, 0, 0, 0);

            device.cmd_end_render_pass(command_buffer);

            device
                .end_command_buffer(command_buffer)
                .expect("Failed to record Command Buffer at Ending!");
        }
    }

    command_buffers
}

pub fn create_descriptor_pool(
    device: &ash::Device,
    swapchain_images_size: usize,
) -> vk::DescriptorPool {
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: swapchain_images_size as u32,
    }];

    let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo {
        s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::DescriptorPoolCreateFlags::empty(),
        max_sets: swapchain_images_size as u32,
        pool_size_count: pool_sizes.len() as u32,
        p_pool_sizes: pool_sizes.as_ptr(),
    };

    unsafe {
        device
            .create_descriptor_pool(&descriptor_pool_create_info, None)
            .expect("Failed to create Descriptor Pool!")
    }
}

pub fn create_descriptor_sets(
    device: &ash::Device,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    uniform_buffers: &Vec<vk::Buffer>,
    swapchain_images_size: usize,
) -> Vec<vk::DescriptorSet> {
    let mut layouts: Vec<vk::DescriptorSetLayout> = vec![];
    for _ in 0..swapchain_images_size {
        layouts.push(descriptor_set_layout);
    }

    let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo {
        s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
        p_next: ptr::null(),
        descriptor_pool,
        descriptor_set_count: swapchain_images_size as u32,
        p_set_layouts: layouts.as_ptr(),
    };

    let descriptor_sets = unsafe {
        device
            .allocate_descriptor_sets(&descriptor_set_allocate_info)
            .expect("Failed to allocate descriptor sets!")
    };

    for (i, &descriptor_set) in descriptor_sets.iter().enumerate() {
        let descriptor_buffer_info = [vk::DescriptorBufferInfo {
            buffer: uniform_buffers[i],
            offset: 0,
            range: std::mem::size_of::<UniformBufferObject>() as u64,
        }];

        let descriptor_write_sets = [vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            p_next: ptr::null(),
            dst_set: descriptor_set,
            dst_binding: 0,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            p_image_info: ptr::null(),
            p_buffer_info: descriptor_buffer_info.as_ptr(),
            p_texel_buffer_view: ptr::null(),
        }];

        unsafe {
            device.update_descriptor_sets(&descriptor_write_sets, &[]);
        }
    }

    descriptor_sets
}

pub fn create_texture_image(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    submit_queue: vk::Queue,
    device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    image_path: &Path,
) -> (vk::Image, vk::DeviceMemory) {
    let mut image_object = image::open(image_path).unwrap();
    image_object = image_object.flipv();
    let (image_width, image_height) = (image_object.width(), image_object.height());
    let image_size =
        (std::mem::size_of::<u8>() as u32 * image_width * image_height * 4) as vk::DeviceSize;
    let image_data = match &image_object {
        image::DynamicImage::ImageLuma8(_) | image::DynamicImage::ImageRgb8(_) => {
            image_object.to_rgba8().into_raw()
        }
        image::DynamicImage::ImageLumaA8(_) | image::DynamicImage::ImageRgba8(_) => {
            image_object.into_bytes()
        }
    };

    if image_size <= 0 {
        panic!("Failed to load texture image!")
    }

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        device,
        image_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        device_memory_properties,
    );

    unsafe {
        let data_ptr = device
            .map_memory(
                staging_buffer_memory,
                0,
                image_size,
                vk::MemoryMapFlags::empty(),
            )
            .expect("Failed to Map Memory") as *mut u8;

        data_ptr.copy_from_nonoverlapping(image_data.as_ptr(), image_data.len());

        device.unmap_memory(staging_buffer_memory);
    }

    let (texture_image, texture_image_memory) = create_image(
        device,
        image_width,
        image_height,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        device_memory_properties,
    );

    transition_image_layout(
        device,
        command_pool,
        submit_queue,
        texture_image,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    );

    copy_buffer_to
}

fn create_image(
    device: &ash::Device,
    width: u32,
    height: u32,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    required_memory_properties: vk::MemoryPropertyFlags,
    device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
) -> (vk::Image, vk::DeviceMemory) {
    let image_create_info = vk::ImageCreateInfo {
        s_type: vk::StructureType::IMAGE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::ImageCreateFlags::empty(),
        image_type: vk::ImageType::TYPE_2D,
        format,
        extent: vk::Extent3D {
            width,
            height,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling,
        usage,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        queue_family_index_count: 0,
        p_queue_family_indices: ptr::null(),
        initial_layout: vk::ImageLayout::UNDEFINED,
    };

    let texture_image = unsafe {
        device
            .create_image(&image_create_info, None)
            .expect("Failed to create Texture Image!")
    };

    let image_memory_requirement = unsafe { device.get_image_memory_requirements(texture_image) };
    let memory_allocate_info = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        p_next: ptr::null(),
        allocation_size: image_memory_requirement.size,
        memory_type_index: find_memory_type(
            image_memory_requirement.memory_type_bits,
            required_memory_properties,
            device_memory_properties,
        ),
    };

    let texture_image_memory = unsafe {
        device
            .allocate_memory(&memory_allocate_info, None)
            .expect("Failed to allocate Texture Image memory!")
    };

    unsafe {
        device
            .bind_image_memory(texture_image, texture_image_memory, 0)
            .expect("Failed to bind Image Memory!");
    }

    (texture_image, texture_image_memory)
}

fn transition_image_layout(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    submit_queue: vk::Queue,
    image: vk::Image,
    _format: vk::Format,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let command_buffer = begin_single_time_command(device, command_pool);

    let src_access_mask;
    let dst_access_mask;
    let source_stage;
    let destination_stage;

    if old_layout == vk::ImageLayout::UNDEFINED
        && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
    {
        src_access_mask = vk::AccessFlags::empty();
        dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
        destination_stage = vk::PipelineStageFlags::TRANSFER;
    } else if old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
        && new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    {
        src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        dst_access_mask = vk::AccessFlags::SHADER_READ;
        source_stage = vk::PipelineStageFlags::TRANSFER;
        destination_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
    } else {
        panic!("Unsupported layout transition!")
    }

    let image_barriers = [vk::ImageMemoryBarrier {
        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
        p_next: ptr::null(),
        src_access_mask,
        dst_access_mask,
        old_layout,
        new_layout,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
    }];

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            source_stage,
            destination_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &image_barriers,
        );
    }

    end_single_time_command(device, command_pool, submit_queue, command_buffer);
}

fn begin_single_time_command(
    device: &ash::Device,
    command_pool: vk::CommandPool,
) -> vk::CommandBuffer {
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        p_next: ptr::null(),
        command_buffer_count: 1,
        command_pool,
        level: vk::CommandBufferLevel::PRIMARY,
    };

    let command_buffer = unsafe {
        device
            .allocate_command_buffers(&command_buffer_allocate_info)
            .expect("Failed to allocate Command Buffers!")
    }[0];

    let command_buffer_begin_info = vk::CommandBufferBeginInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
        p_next: ptr::null(),
        p_inheritance_info: ptr::null(),
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
    };

    unsafe {
        device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)
            .expect("Failed to begin recording Command Buffer at beginning!");
    }

    command_buffer
}

fn end_single_time_command(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    submit_queue: vk::Queue,
    command_buffer: vk::CommandBuffer,
) {
    unsafe {
        device
            .end_command_buffer(command_buffer)
            .expect("Failed to record Command Buffer at Ending!");
    }

    let buffers_to_submit = [command_buffer];

    let submit_infos = [vk::SubmitInfo {
        s_type: vk::StructureType::SUBMIT_INFO,
        p_next: ptr::null(),
        wait_semaphore_count: 0,
        p_wait_semaphores: ptr::null(),
        p_wait_dst_stage_mask: ptr::null(),
        command_buffer_count: 1,
        p_command_buffers: buffers_to_submit.as_ptr(),
        signal_semaphore_count: 0,
        p_signal_semaphores: ptr::null(),
    }];

    unsafe {
        device
            .queue_submit(submit_queue, &submit_infos, vk::Fence::null())
            .expect("Failed to Queue Submit!");
        device
            .queue_wait_idle(submit_queue)
            .expect("Failed to wait Queue idle!");
        device.free_command_buffers(command_pool, &buffers_to_submit);
    }
}

fn copy_buffer_to_image(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    submit_queue: vk::Queue,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) {
    todo!()
}

pub fn create_sync_objects(device: &ash::Device, max_frames_in_flight: usize) -> SyncObjects {
    let mut sync_objects = SyncObjects {
        image_available_semaphores: vec![],
        render_finished_semaphores: vec![],
        inflight_fences: vec![],
    };

    let semaphore_create_info = vk::SemaphoreCreateInfo {
        s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::SemaphoreCreateFlags::empty(),
    };

    let fence_create_info = vk::FenceCreateInfo {
        s_type: vk::StructureType::FENCE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::FenceCreateFlags::SIGNALED,
    };

    for _ in 0..max_frames_in_flight {
        unsafe {
            let image_available_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .expect("Failed to create Semaphore Object!");
            let render_finished_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .expect("Failed to create Semaphore Object");
            let inflight_fence = device
                .create_fence(&fence_create_info, None)
                .expect("Failed to create Fence Object!");

            sync_objects
                .image_available_semaphores
                .push(image_available_semaphore);
            sync_objects
                .render_finished_semaphores
                .push(render_finished_semaphore);
            sync_objects.inflight_fences.push(inflight_fence);
        }
    }

    sync_objects
}
