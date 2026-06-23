from utils.memory import get_memory_manager

m = get_memory_manager(0.25, 0.45)
print('Memory Manager initialized successfully')
stats = m.get_memory_stats()
print(f'RAM: {stats["ram"]["process_rss_gb"]:.2f}GB ({stats["ram"]["process_rss_ratio"]*100:.1f}%)')
print('Memory management module test passed')
