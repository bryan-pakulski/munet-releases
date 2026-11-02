// Zero-copy Vulkan interop through the C API: a separate component (this test) shares MuNet's VkDevice, writes the
// program's inputs into MuNet's buffer with its own command buffer, MuNet runs ordered by a timeline semaphore, and the
// component copies the outputs out, again GPU-ordered. The host waits once, at the end.
//
//   vulkan_interop <tiny.mnet> [video]      (rust/tests/data/tiny.mnet: y = x @ W + k)
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include "munet/munet_c.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { std::printf("FAIL %s: VkResult %d\n", #x, r_); return 1; } } while (0)
#define REQUIRE(c, msg) do { if (!(c)) { std::printf("FAIL %s: %s\n", msg, munet_last_error()); return 1; } } while (0)

int main(int argc, char** argv) {
  if (argc < 2) { std::printf("usage: vulkan_interop tiny.mnet [video]\n"); return 2; }
  const bool video = argc > 2 && std::strcmp(argv[2], "video") == 0;
  munet_vulkan* ctx = munet_vulkan_create(0, video ? MUNET_VULKAN_VIDEO_DECODE : 0);
  REQUIRE(ctx, "munet_vulkan_create");
  munet_vulkan_handles h{};
  REQUIRE(munet_vulkan_get_handles(ctx, &h) == 0, "handles");
  std::printf("api %u.%u, compute family %u, decode family %d (%u queues), %u extensions, timeline %d\n",
              VK_API_VERSION_MAJOR(h.api_version), VK_API_VERSION_MINOR(h.api_version), h.compute_family,
              int(h.decode_family), h.decode_queues, h.extension_count, h.timeline_semaphores);
  if (video) REQUIRE(h.decode_queues > 0, "video decode queue");

  auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(h.get_instance_proc_addr);
  auto instance = static_cast<VkInstance>(h.instance);
  auto device = static_cast<VkDevice>(h.device);
  auto gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(gipa(instance, "vkGetDeviceProcAddr"));
#define DEV(name) auto name = reinterpret_cast<PFN_##name>(gdpa(device, #name)); if (!name) { std::printf("missing %s\n", #name); return 1; }
#define INST(name) auto name = reinterpret_cast<PFN_##name>(gipa(instance, #name));
  INST(vkGetPhysicalDeviceMemoryProperties)
  DEV(vkGetDeviceQueue) DEV(vkCreateBuffer) DEV(vkGetBufferMemoryRequirements) DEV(vkAllocateMemory) DEV(vkBindBufferMemory)
  DEV(vkMapMemory) DEV(vkCreateCommandPool) DEV(vkAllocateCommandBuffers) DEV(vkBeginCommandBuffer) DEV(vkEndCommandBuffer)
  DEV(vkCmdCopyBuffer) DEV(vkCmdPipelineBarrier) DEV(vkQueueSubmit) DEV(vkCreateSemaphore) DEV(vkWaitSemaphores)
  DEV(vkDestroySemaphore) DEV(vkDestroyCommandPool) DEV(vkDestroyBuffer) DEV(vkFreeMemory)

  munet_options o = munet_default_options();
  o.precision = MUNET_PRECISION_FP32;
  munet_program* p = munet_load_on(ctx, argv[1], &o);
  REQUIRE(p, "munet_load_on");
  auto program_buffer = static_cast<VkBuffer>(munet_program_vulkan_buffer(p));
  REQUIRE(program_buffer, "program buffer");

  // The component's own host-visible buffer: inputs x (6), k (2) then room for y (8).
  VkPhysicalDeviceMemoryProperties mem;
  vkGetPhysicalDeviceMemoryProperties(static_cast<VkPhysicalDevice>(h.physical_device), &mem);
  VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = 16 * sizeof(float);
  bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VkBuffer host = VK_NULL_HANDLE;
  CHECK(vkCreateBuffer(device, &bi, nullptr, &host));
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(device, host, &req);
  uint32_t type = UINT32_MAX;
  for (uint32_t i = 0; i < mem.memoryTypeCount && type == UINT32_MAX; ++i)
    if ((req.memoryTypeBits & (1u << i)) && (mem.memoryTypes[i].propertyFlags &
        (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
        (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) type = i;
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size; ai.memoryTypeIndex = type;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  CHECK(vkAllocateMemory(device, &ai, nullptr, &memory));
  CHECK(vkBindBufferMemory(device, host, memory, 0));
  float* data = nullptr;
  CHECK(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void**>(&data)));
  const float x[6] = {1, 2, 3, -1, 0.5f, 2}, k[2] = {10, -3}, expect[8] = {14, 15, 9, 15, -2, -0.5f, -7, -0.5f};
  std::memcpy(data, x, sizeof x);
  std::memcpy(data + 6, k, sizeof k);

  VkSemaphoreTypeCreateInfo st{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  st.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  sci.pNext = &st;
  VkSemaphore timeline = VK_NULL_HANDLE;
  CHECK(vkCreateSemaphore(device, &sci, nullptr, &timeline));

  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.queueFamilyIndex = h.compute_family;
  VkCommandPool pool = VK_NULL_HANDLE;
  CHECK(vkCreateCommandPool(device, &pci, nullptr, &pool));
  VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 2;
  VkCommandBuffer cmd[2];
  CHECK(vkAllocateCommandBuffers(device, &cai, cmd));
  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
  // Upload: the component's buffer -> the program's input regions.
  CHECK(vkBeginCommandBuffer(cmd[0], &begin));
  VkBufferCopy in[2] = {{0, VkDeviceSize(munet_input_offset(p, 0)), sizeof x},
                        {sizeof x, VkDeviceSize(munet_input_offset(p, 1)), sizeof k}};
  vkCmdCopyBuffer(cmd[0], host, program_buffer, 2, in);
  vkCmdPipelineBarrier(cmd[0], VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
  CHECK(vkEndCommandBuffer(cmd[0]));
  // Download: the program's output y -> the component's buffer.
  CHECK(vkBeginCommandBuffer(cmd[1], &begin));
  VkBufferCopy out{VkDeviceSize(munet_output_offset(p, 0)), 8 * sizeof(float), 8 * sizeof(float)};
  vkCmdCopyBuffer(cmd[1], program_buffer, host, 1, &out);
  barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(cmd[1], VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
  CHECK(vkEndCommandBuffer(cmd[1]));

  VkQueue queue = VK_NULL_HANDLE;
  vkGetDeviceQueue(device, h.compute_family, h.compute_queue, &queue);
  auto submit = [&](VkCommandBuffer c, uint64_t wait, uint64_t signal) {
    VkTimelineSemaphoreSubmitInfo ts{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.pNext = &ts; si.commandBufferCount = 1; si.pCommandBuffers = &c;
    if (wait) { si.waitSemaphoreCount = 1; si.pWaitSemaphores = &timeline; si.pWaitDstStageMask = &stage;
                ts.waitSemaphoreValueCount = 1; ts.pWaitSemaphoreValues = &wait; }
    si.signalSemaphoreCount = 1; si.pSignalSemaphores = &timeline; ts.signalSemaphoreValueCount = 1; ts.pSignalSemaphoreValues = &signal;
    munet_vulkan_lock_queue(ctx, h.compute_family, h.compute_queue);  // MuNet submits to the same queue
    VkResult r = vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    munet_vulkan_unlock_queue(ctx, h.compute_family, h.compute_queue);
    return r;
  };
  double worst = 0;
  for (uint64_t round = 0; round < 3; ++round) {
    const uint64_t base = round * 3;
    std::memset(data + 8, 0, 8 * sizeof(float));
    CHECK(submit(cmd[0], base, base + 1));                  // inputs written -> base+1
    munet_semaphore_sync sync{timeline, base + 1, timeline, base + 2};
    REQUIRE(munet_run_device(p, &sync) == 0, "munet_run_device");   // waits base+1, signals base+2
    CHECK(submit(cmd[1], base + 2, base + 3));              // outputs read -> base+3
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    const uint64_t done = base + 3;
    wi.semaphoreCount = 1; wi.pSemaphores = &timeline; wi.pValues = &done;
    CHECK(vkWaitSemaphores(device, &wi, UINT64_MAX));
    for (int i = 0; i < 8; ++i) worst = std::fmax(worst, std::fabs(data[8 + i] - expect[i]));
  }
  REQUIRE(munet_wait(p) == 0, "munet_wait");
  std::printf("y = [%g %g %g %g %g %g %g %g]; max |error| over 3 GPU-ordered rounds: %g\n", data[8], data[9], data[10],
              data[11], data[12], data[13], data[14], data[15], worst);
  vkDestroyCommandPool(device, pool, nullptr);
  vkDestroySemaphore(device, timeline, nullptr);
  vkDestroyBuffer(device, host, nullptr);
  vkFreeMemory(device, memory, nullptr);
  munet_free(p);
  munet_vulkan_free(ctx);
  if (worst > 1e-5) { std::printf("FAIL: wrong outputs\n"); return 1; }
  std::printf("OK\n");
  return 0;
}
