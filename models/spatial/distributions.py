# models/spatial/distributions.py

import numpy as np
from sklearn.datasets import make_blobs, make_circles
from typing import Dict, Any

def create_spatial_distribution(num_nodes: int, dist_config: Dict[str, Any]) -> np.ndarray:
    """
    Generates spatial coordinates for nodes within a normalized [0, 1] x [0, 1] space.
    All parameters for the distribution are passed via the dist_config dictionary, 
    which is intended to be loaded from a YAML configuration file.

    Args:
        num_nodes (int): The total number of nodes to generate coordinates for.
        dist_config (Dict[str, Any]): A dictionary containing the configuration for the distribution.
                                      It must include a 'type' key specifying the distribution name.

    Returns:
        np.ndarray: A numpy array of shape (num_nodes, 2) containing the [x, y] coordinates for each node.
        
    Raises:
        ValueError: If the specified distribution type is unknown or parameters are missing/invalid.
    """
    dist_type = dist_config.get('type', 'uniform').lower()
    
    locations = None

    if dist_type == 'uniform':
        # --- Uniform Distribution ---
        # Distributes nodes uniformly within the [0, 1] x [0, 1] square.
        #
        # Corresponding YAML config structure:
        #
        # spatial:
        #   distribution:
        #     type: uniform
        #
        # Note: 'low' and 'high' parameters are not needed as they are fixed to [0,0] and [1,1].
        low = [0.0, 0.0]
        high = [1.0, 1.0]
        locations = np.random.uniform(low=low, high=high, size=(num_nodes, 2))

    elif dist_type == 'gaussian':
        # --- Gaussian (Normal) Distribution ---
        # Distributes nodes around a central point. For the distribution to be mostly
        # within the [0,1]x[0,1] area, mean should be near [0.5, 0.5] and std_dev should be small.
        #
        # Corresponding YAML config structure:
        #
        # spatial:
        #   distribution:
        #     type: gaussian
        #     mean: [0.5, 0.5]
        #     std_dev: [0.15, 0.15]
        #
        mean = dist_config.get('mean', [0.5, 0.5])
        std_dev = dist_config.get('std_dev', [0.15, 0.15])
        cov = np.diag(np.square(std_dev))
        locations = np.random.multivariate_normal(mean=mean, cov=cov, size=num_nodes)
        # Clip the values to ensure they stay within the [0, 1] bounds
        locations = np.clip(locations, 0.0, 1.0)
    
    elif dist_type == 'satellite':
        # --- Satellite Distribution ---
        # Models a main central city with one or more smaller satellite cities.
        # This is a composite of multiple Gaussian distributions with different node counts.
        #
        # Corresponding YAML config structure:
        #
        # spatial:
        #   distribution:
        #     type: satellite
        #     main_city:
        #       center: [0.5, 0.5]
        #       proportion: 0.7  # 70% of nodes are in the main city
        #       std_dev: 0.1
        #     satellites:
        #       - center: [0.1, 0.8]
        #         proportion: 0.15 # 15% of nodes
        #         std_dev: 0.05
        #       - center: [0.9, 0.2]
        #         proportion: 0.15 # 15% of nodes
        #         std_dev: 0.05
        #
        main_city_config = dist_config.get('main_city')
        satellites_config = dist_config.get('satellites', [])
        
        if not main_city_config:
            raise ValueError("'satellite' distribution requires a 'main_city' configuration.")
            
        all_locations = []
        assigned_nodes = 0
        
        # Calculate node counts for each satellite
        for sat_conf in satellites_config:
            num_sat_nodes = int(num_nodes * sat_conf['proportion'])
            mean = sat_conf['center']
            std_dev = sat_conf['std_dev']
            cov = np.diag(np.square(std_dev))
            sat_locs = np.random.multivariate_normal(mean=mean, cov=cov, size=num_sat_nodes)
            all_locations.append(sat_locs)
            assigned_nodes += num_sat_nodes
            
        # Main city gets the remaining nodes
        num_main_nodes = num_nodes - assigned_nodes
        mean = main_city_config['center']
        std_dev = main_city_config['std_dev']
        cov = np.diag(np.square(std_dev))
        main_locs = np.random.multivariate_normal(mean=mean, cov=cov, size=num_main_nodes)
        all_locations.append(main_locs)
        
        locations = np.vstack(all_locations)
        np.random.shuffle(locations) # Shuffle to mix nodes from different cities
        locations = np.clip(locations, 0.0, 1.0)

    elif dist_type == 'clustered':
        # --- Clustered Distribution ---
        # Creates one or more distinct clusters of nodes.
        #
        # Corresponding YAML config structure (option 1: random centers):
        #
        # spatial:
        #   distribution:
        #     type: clustered
        #     n_clusters: 4
        #     cluster_std: 0.05
        #
        # Corresponding YAML config structure (option 2: explicit centers):
        #
        # spatial:
        #   distribution:
        #     type: clustered
        #     centers: [[0.2, 0.2], [0.8, 0.8], [0.2, 0.8]]
        #     cluster_std: 0.05
        #
        n_clusters = dist_config.get('n_clusters', 3)
        cluster_std = dist_config.get('cluster_std', 0.05)
        centers = dist_config.get('centers', None)
        # The center_box for sklearn's make_blobs is fixed to the unit square
        center_box = (0.0, 1.0, 0.0, 1.0) 

        if centers is None:
            locations, _ = make_blobs(n_samples=num_nodes, n_features=2, centers=n_clusters,
                                      cluster_std=cluster_std, center_box=center_box)
        else:
            locations, _ = make_blobs(n_samples=num_nodes, n_features=2, centers=np.array(centers),
                                      cluster_std=cluster_std)
        # Scale and shift the locations to ensure they are within [0, 1] x [0, 1]
        loc_min, loc_max = locations.min(axis=0), locations.max(axis=0)
        locations = (locations - loc_min) / (loc_max - loc_min)


    elif dist_type == 'grid':
        # --- Grid Distribution ---
        # Places nodes in a regular grid pattern over the unit square.
        #
        # Corresponding YAML config structure:
        #
        # spatial:
        #   distribution:
        #     type: grid
        #     jitter: 0.01 # Optional: adds random noise to each point
        #
        jitter = dist_config.get('jitter', 0.0)
        
        n_side = int(np.ceil(np.sqrt(num_nodes)))
        x = np.linspace(0.0, 1.0, n_side)
        y = np.linspace(0.0, 1.0, n_side)
        
        xv, yv = np.meshgrid(x, y)
        locations = np.vstack([xv.ravel(), yv.ravel()]).T
        locations = locations[:num_nodes] # Trim excess points
        
        if jitter > 0:
            locations += np.random.uniform(-jitter, jitter, size=locations.shape)
            locations = np.clip(locations, 0.0, 1.0) # Ensure jitter does not push points out of bounds
    
    elif dist_type == 'segregated':
        # --- Segregated Distribution ---
        # Models distinct regions (e.g., rich/poor areas) separated by a boundary.
        # Nodes are distributed uniformly within their assigned rectangular region.
        #
        # Corresponding YAML config structure:
        #
        # spatial:
        #   distribution:
        #     type: segregated
        #     regions:
        #       - proportion: 0.5 # 50% of nodes
        #         bounds: [0.0, 0.45, 0.0, 1.0] # [xmin, xmax, ymin, ymax] for area 1
        #       - proportion: 0.5 # 50% of nodes
        #         bounds: [0.55, 1.0, 0.0, 1.0] # [xmin, xmax, ymin, ymax] for area 2
        #
        regions = dist_config.get('regions')
        if not regions:
            raise ValueError("'segregated' distribution requires a 'regions' configuration.")
            
        all_locations = []
        remaining_nodes = num_nodes
        
        for i, region in enumerate(regions):
            if i == len(regions) - 1:
                # Assign all remaining nodes to the last region to avoid rounding errors
                num_region_nodes = remaining_nodes
            else:
                num_region_nodes = int(num_nodes * region['proportion'])
                remaining_nodes -= num_region_nodes
            
            bounds = region['bounds']
            low = [bounds[0], bounds[2]]
            high = [bounds[1], bounds[3]]
            
            region_locs = np.random.uniform(low=low, high=high, size=(num_region_nodes, 2))
            all_locations.append(region_locs)
            
        locations = np.vstack(all_locations)
        np.random.shuffle(locations)
        locations = np.clip(locations, 0.0, 1.0)

    elif dist_type == 'linear':
        # --- Linear Distribution ---
        # Places nodes along a straight line within the unit square.
        #
        # Corresponding YAML config structure:
        #
        # spatial:
        #   distribution:
        #     type: linear
        #     start: [0.1, 0.1]
        #     end: [0.9, 0.9]
        #     noise: 0.02 # Optional: Perpendicular noise/spread
        #
        start = np.array(dist_config.get('start', [0.0, 0.0]))
        end = np.array(dist_config.get('end', [1.0, 1.0]))
        noise = dist_config.get('noise', 0.0)

        t = np.linspace(0, 1, num_nodes)[:, np.newaxis]
        locations = start + t * (end - start)

        if noise > 0:
            direction_vec = end - start
            # Handle the case where the vector is zero
            norm = np.linalg.norm(direction_vec)
            if norm > 1e-9:
                perp_vec = np.array([-direction_vec[1], direction_vec[0]]) / norm
                noise_vec = np.random.normal(0, noise, size=(num_nodes, 1)) * perp_vec
                locations += noise_vec
        locations = np.clip(locations, 0.0, 1.0)

    elif dist_type == 'crossroads':
        # --- Crossroads Distribution ---
        # Models nodes concentrated along two or more intersecting linear paths (roads).
        # This is a composite of multiple linear distributions.
        #
        # Corresponding YAML config structure:
        #
        # spatial:
        #   distribution:
        #     type: crossroads
        #     roads:
        #       - proportion: 0.5 # 50% of nodes on the horizontal road
        #         start: [0.0, 0.5]
        #         end: [1.0, 0.5]
        #         noise: 0.02
        #       - proportion: 0.5 # 50% of nodes on the vertical road
        #         start: [0.5, 0.0]
        #         end: [0.5, 1.0]
        #         noise: 0.02
        #
        roads = dist_config.get('roads')
        if not roads:
            raise ValueError("'crossroads' distribution requires a 'roads' configuration.")
            
        all_locations = []
        remaining_nodes = num_nodes
        
        for i, road in enumerate(roads):
            if i == len(roads) - 1:
                num_road_nodes = remaining_nodes
            else:
                num_road_nodes = int(num_nodes * road['proportion'])
                remaining_nodes -= num_road_nodes
            
            start = np.array(road['start'])
            end = np.array(road['end'])
            noise = road.get('noise', 0.0)
            
            t = np.linspace(0, 1, num_road_nodes)[:, np.newaxis]
            road_locs = start + t * (end - start)
            
            if noise > 0:
                direction_vec = end - start
                norm = np.linalg.norm(direction_vec)
                if norm > 1e-9:
                    perp_vec = np.array([-direction_vec[1], direction_vec[0]]) / norm
                    noise_vec = np.random.normal(0, noise, size=(num_road_nodes, 1)) * perp_vec
                    road_locs += noise_vec
            
            all_locations.append(road_locs)
            
        locations = np.vstack(all_locations)
        np.random.shuffle(locations)
        locations = np.clip(locations, 0.0, 1.0)

    elif dist_type == 'circles':
        # --- Concentric Circles Distribution ---
        # Places nodes on two concentric circles, scaled to fit the unit square.
        #
        # Corresponding YAML config structure:
        #
        # spatial:
        #   distribution:
        #     type: circles
        #     factor: 0.5 # Scale factor between inner and outer circle (0 < factor < 1)
        #     noise: 0.05 # Standard deviation of Gaussian noise
        #
        factor = dist_config.get('factor', 0.5)
        noise = dist_config.get('noise', 0.05)
        # make_circles generates data in approx [-1, 1] range centered at (0,0)
        locations, _ = make_circles(n_samples=num_nodes, factor=factor, noise=noise)
        # Scale and shift to [0, 1] range
        locations = (locations + 1.0) / 2.0
        locations = np.clip(locations, 0.0, 1.0)


    elif dist_type == 'wedge':
        # --- Wedge Distribution ---
        # Distributes nodes within a sector of an annulus (a wedge), centered in the unit square.
        #
        # Corresponding YAML config structure:
        #
        # spatial:
        #   distribution:
        #     type: wedge
        #     radius_range: [0.2, 0.5] # Min and max radius. Max should be <= 0.5.
        #     angle_range: [0, 90]    # Start and end angle in degrees
        #
        center = np.array([0.5, 0.5]) # Center is fixed to the middle of the unit square
        r_range = dist_config.get('radius_range', [0.0, 0.5])
        angle_range_deg = dist_config.get('angle_range', [0, 360])
        
        angle_range_rad = np.deg2rad(angle_range_deg)
        
        # Sample r^2 uniformly then take sqrt for uniform spatial density
        r_squared = np.random.uniform(r_range[0]**2, r_range[1]**2, num_nodes)
        r = np.sqrt(r_squared)
        theta = np.random.uniform(angle_range_rad[0], angle_range_rad[1], num_nodes)
        
        x = r * np.cos(theta) + center[0]
        y = r * np.sin(theta) + center[1]
        locations = np.vstack([x, y]).T
        locations = np.clip(locations, 0.0, 1.0)

    elif dist_type == 'hex_lattice':
        # --- Hexagonal Lattice Distribution (Central Place Theory) ---
        # Places nodes on a hexagonal grid, which is the most efficient way to tile a plane.
        # This is inspired by Christaller's Central Place Theory for settlement distribution.
        #
        # Corresponding YAML config structure:
        #
        # spatial:
        #   distribution:
        #     type: hex_lattice
        #     jitter: 0.01  # Optional: adds random noise around each lattice point
        #
        jitter = dist_config.get('jitter', 0.0)

        # Estimate the number of hexagons needed to cover the area for num_nodes
        # Area of unit square is 1. Area of a hexagon is (3*sqrt(3)/2)*s^2.
        # We need num_nodes points. So area per point is 1/num_nodes.
        # Set area of hexagon ~ area per point to find side length s.
        # This is an approximation to determine grid density.
        num_hex_y = int(np.ceil(np.sqrt(num_nodes / (np.sqrt(3)))))
        num_hex_x = int(np.ceil(num_nodes / num_hex_y))
        
        dy = 1.0 / num_hex_y
        dx = 1.0 / num_hex_x
        s = dx # side length approximation
        
        points = []
        for j in range(num_hex_y * 2): # Iterate over more rows to be safe
            for i in range(num_hex_x * 2): # And more columns
                x = i * dx + (j % 2) * (dx / 2)
                y = j * (np.sqrt(3) / 2 * s) # Use side length for y-spacing
                if x <= 1.05 and y <= 1.05: # Add a small buffer
                    points.append([x, y])
        
        locations = np.array(points)
        # Scale to fit exactly within [0,1]x[0,1]
        if locations.shape[0] > 0:
            locations -= locations.min(axis=0)
            locations /= locations.max(axis=0)
        
        if locations.shape[0] < num_nodes:
             raise ValueError(f"Hex lattice failed to generate enough points. Try adjusting logic.")
             
        # Take the first num_nodes points and shuffle them
        np.random.shuffle(locations)
        locations = locations[:num_nodes]
        
        if jitter > 0:
            locations += np.random.uniform(-jitter, jitter, size=locations.shape)
            
        locations = np.clip(locations, 0.0, 1.0)

    elif dist_type == 'concentric_rings':
        # --- Concentric Rings Distribution (Burgess Model) ---
        # Models a city as a series of concentric rings, each with a different node proportion.
        # This is inspired by the Concentric Zone Model of urban structure.
        #
        # Corresponding YAML config structure:
        #
        # spatial:
        #   distribution:
        #     type: concentric_rings
        #     center: [0.5, 0.5]
        #     rings:
        #       - proportion: 0.1 # 10% of nodes in the CBD
        #         radius_range: [0.0, 0.05]
        #       - proportion: 0.3 # 30% in the next zone
        #         radius_range: [0.05, 0.2]
        #       - proportion: 0.6 # 60% in the suburbs
        #         radius_range: [0.2, 0.5]
        #
        center = np.array(dist_config.get('center', [0.5, 0.5]))
        rings = dist_config.get('rings')
        if not rings:
            raise ValueError("'concentric_rings' distribution requires a 'rings' configuration.")
            
        all_locations = []
        remaining_nodes = num_nodes
        
        for i, ring in enumerate(rings):
            if i == len(rings) - 1:
                num_ring_nodes = remaining_nodes
            else:
                num_ring_nodes = int(num_nodes * ring['proportion'])
                remaining_nodes -= num_ring_nodes
            
            r_range = ring['radius_range']
            
            # Sample r^2 uniformly then take sqrt for uniform spatial density within the annulus
            r_squared = np.random.uniform(r_range[0]**2, r_range[1]**2, num_ring_nodes)
            r = np.sqrt(r_squared)
            theta = np.random.uniform(0, 2 * np.pi, num_ring_nodes)
            
            x = r * np.cos(theta) + center[0]
            y = r * np.sin(theta) + center[1]
            all_locations.append(np.vstack([x, y]).T)
            
        locations = np.vstack(all_locations)
        np.random.shuffle(locations)
        locations = np.clip(locations, 0.0, 1.0)

    elif dist_type == 'radial_decay':
        # --- Radial Decay Distribution (Urban Sprawl) ---
        # Models a distribution where node density decreases with distance from a central point.
        # This can simulate phenomena like urban sprawl.
        #
        # Corresponding YAML config structure:
        #
        # spatial:
        #   distribution:
        #     type: radial_decay
        #     center: [0.5, 0.5]
        #     max_radius: 0.5 # The maximum extent of the sprawl
        #     strength: 2.0   # Controls how quickly density falls off.
        #                     # strength=1.0 is uniform, >1.0 decays, <1.0 concentrates at edge.
        #                     # strength=2.0 gives a linear decay of density.
        #
        center = np.array(dist_config.get('center', [0.5, 0.5]))
        max_radius = dist_config.get('max_radius', 0.5)
        strength = dist_config.get('strength', 2.0)
        
        # Generate angles uniformly
        theta = np.random.uniform(0, 2 * np.pi, num_nodes)
        
        # Generate radii non-uniformly using inverse transform sampling
        # A strength of 2.0 corresponds to a probability density function p(r) proportional to (R-r)
        u = np.random.uniform(0, 1, num_nodes)
        r = max_radius * (1 - u**(1.0 / strength))
        
        x = r * np.cos(theta) + center[0]
        y = r * np.sin(theta) + center[1]
        
        locations = np.vstack([x, y]).T
        locations = np.clip(locations, 0.0, 1.0)

    else:
        raise ValueError(f"Unknown spatial distribution type: '{dist_type}'")

    return locations

