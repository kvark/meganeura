use ash::{vk, Entry};
use std::{ffi::{c_void, CStr}, ptr};

#[repr(C)]
struct Flexible {
    s_type: vk::StructureType, next: *mut c_void,
    m: u32, n: u32, k: u32,
    a: vk::ComponentTypeKHR, b: vk::ComponentTypeKHR,
    c: vk::ComponentTypeKHR, result: vk::ComponentTypeKHR,
    saturating: u32, scope: vk::ScopeKHR, invocations: u32,
}

fn main() {
    unsafe {
        let entry = Entry::load().unwrap();
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance = entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None).unwrap();
        let coop = ash::khr::cooperative_matrix::Instance::new(&entry, &instance);
        for device in instance.enumerate_physical_devices().unwrap() {
            let info = instance.get_physical_device_properties(device);
            println!("DEVICE {} id={:#x}", CStr::from_ptr(info.device_name.as_ptr()).to_str().unwrap(), info.device_id);
            let extensions = instance.enumerate_device_extension_properties(device).unwrap();
            for ext in &extensions {
                let name = CStr::from_ptr(ext.extension_name.as_ptr()).to_str().unwrap();
                if name.contains("cooperative") { println!("EXT {name}"); }
            }
            if !extensions.iter().any(|p| CStr::from_ptr(p.extension_name.as_ptr()) == ash::khr::cooperative_matrix::NAME) { continue; }
            for p in coop.get_physical_device_cooperative_matrix_properties(device).unwrap() {
                println!("KHR {}x{}x{} {:?}/{:?}/{:?}/{:?} {:?} sat={}", p.m_size, p.n_size, p.k_size, p.a_type, p.b_type, p.c_type, p.result_type, p.scope, p.saturating_accumulation);
            }
            if extensions.iter().any(|p| CStr::from_ptr(p.extension_name.as_ptr()) == c"VK_NV_cooperative_matrix2") {
                let address = entry.get_instance_proc_addr(instance.handle(), c"vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV".as_ptr()).unwrap();
                let query: unsafe extern "system" fn(vk::PhysicalDevice, *mut u32, *mut Flexible) -> vk::Result = std::mem::transmute(address);
                let mut count = 0;
                assert_eq!(query(device, &mut count, ptr::null_mut()), vk::Result::SUCCESS);
                let mut rows: Vec<Flexible> = (0..count).map(|_| Flexible {
                    s_type: vk::StructureType::from_raw(1000593001), next: ptr::null_mut(),
                    m: 0, n: 0, k: 0, a: vk::ComponentTypeKHR::FLOAT32, b: vk::ComponentTypeKHR::FLOAT32,
                    c: vk::ComponentTypeKHR::FLOAT32, result: vk::ComponentTypeKHR::FLOAT32,
                    saturating: 0, scope: vk::ScopeKHR::SUBGROUP, invocations: 0,
                }).collect();
                assert_eq!(query(device, &mut count, rows.as_mut_ptr()), vk::Result::SUCCESS);
                for p in rows {
                    println!("NV2 {}x{}x{} {:?}/{:?}/{:?}/{:?} {:?} threads={} sat={}", p.m, p.n, p.k, p.a, p.b, p.c, p.result, p.scope, p.invocations, p.saturating);
                }
            }
        }
        instance.destroy_instance(None);
    }
}
