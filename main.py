import bing_image_downloader as bim
queries = ['hotdog', 'tasty hotdog']
for query in queries:
    bim.download('query', limit=100,  output_dir='dataset',
                        adult_filter_off=True, force_replace=False, timeout=60, verbose=True)
