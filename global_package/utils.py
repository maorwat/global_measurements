def parse_beam_plane(option):
    
    # Map to determine beam
    beam_map = {'B1': 'b1', 'B2': 'b2'}
    # Map to determine plane
    plane_map = {'H': 'horizontal', 'V': 'vertical'}
    
    # Extract beam and plane parts
    beam_key = option[:2]  # First two characters
    plane_key = option[2]  # Last character
    
    # Look up beam and plane
    beam = beam_map.get(beam_key.upper())
    plane = plane_map.get(plane_key.upper())
    
    if beam and plane:
        return beam, plane
    else:
        raise ValueError("Invalid option. Valid options are: B1H, B1V, B2H, B2V.")
        
        
def get_utc_time(start_time, end_time):
    
    # Localize the timestamps to Swiss time (Europe/Zurich)
    tStart = start_time.tz_localize('Europe/Zurich')  # Localize to Swiss time (CET/CEST)
    tEnd = end_time.tz_localize('Europe/Zurich')

    # Convert to UTC
    tStart_utc = tStart.astimezone('UTC')
    tEnd_utc = tEnd.astimezone('UTC')
    
    return tStart_utc, tEnd_utc