"""
Massachusetts Building Data Processor - Enhanced Version with Multi-dimensional Clustering and Soil Analysis
This script processes the building data and exports it to JSON for the updated web dashboard
Now includes pre-computed clustering results for different feature combinations and soil analysis
Split file version to handle GitHub 25MB limit
Fixed: Proper handling of NaN values for JSON export
Enhanced: Added compname analysis and data flow statistics
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import json
from datetime import datetime
import warnings
import re
from collections import defaultdict, Counter

warnings.filterwarnings('ignore')

def format_large_number(num, is_area=False):
    """Turn big numbers to readable numbers"""
    if num >= 1000000:
        return f"{num / 1000000:.2f}M"
    if num >= 1000:
        return f"{num / 1000:.2f}K" if is_area else f"{num / 1000:.1f}K"
    return str(round(num)) if is_area else str(num)


class BuildingDataProcessor:
    def __init__(self, csv_path='ma_structures_with_demolition_FINAL.csv'):
        """Initialize the processor with data path"""
        self.csv_path = csv_path
        self.df = None
        self.df_cleaned = None
        self.df_cluster = None
        self.preprocessor = None
        self.kmeans = None
        self.data_flow_stats = {}  # Track data flow statistics

    def load_data(self):
        """Load the CSV data"""
        print("Loading data...")
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} records")

        # Track initial data stats
        self.data_flow_stats['initial_count'] = len(self.df)
        self.data_flow_stats['initial_columns'] = list(self.df.columns)

        return self

    def clean_data(self):
        """Clean the data with detailed tracking"""
        print("Cleaning data...")

        # Initialize detailed cleaning statistics
        cleaning_stats = {
            'initial_count': len(self.df),
            'initial_columns': list(self.df.columns)
        }

        # Step 1: Track invalid year_built (includes <= 0 and NaN)
        # Note: self.df['year_built'] > 0 returns False for both <= 0 and NaN values
        valid_year_mask = self.df['year_built'] > 0
        cleaning_stats['invalid_year_count'] = (~valid_year_mask).sum()
        cleaning_stats['invalid_year_details'] = {
            'negative_or_zero': (self.df['year_built'] <= 0).sum(),
            'nan_values': self.df['year_built'].isna().sum()
        }

        # Apply year filter using original logic (removes both invalid and NaN)
        self.df_cleaned = self.df[self.df['year_built'] > 0].copy()
        cleaning_stats['after_year_filter'] = len(self.df_cleaned)

        # Step 2: Track and remove missing OR zero/negative area
        invalid_area_mask = (self.df_cleaned['Est GFA sqmeters'].isna()) | (self.df_cleaned['Est GFA sqmeters'] <= 0)
        cleaning_stats['missing_area_count'] = invalid_area_mask.sum()

        # Remove rows with invalid area
        if cleaning_stats['missing_area_count'] > 0:
            self.df_cleaned = self.df_cleaned[~invalid_area_mask].copy()
        cleaning_stats['after_missing_area'] = len(self.df_cleaned)

        # Step 4: Track and remove area outliers (optional step)
        cleaning_stats['area_outliers_count'] = 0
        cleaning_stats['area_outlier_threshold'] = None

        # Check if we have valid area data to calculate outliers
        if 'Est GFA sqmeters' in self.df_cleaned.columns and len(self.df_cleaned) > 0:
            # Calculate the 99.999th percentile for outlier detection
            area_threshold = self.df_cleaned['Est GFA sqmeters'].quantile(0.99999)
            outlier_mask = self.df_cleaned['Est GFA sqmeters'] > area_threshold
            cleaning_stats['area_outliers_count'] = outlier_mask.sum()
            cleaning_stats['area_outlier_threshold'] = float(area_threshold)

            # COMMENTED OUT: Don't remove outliers anymore
            # if cleaning_stats['area_outliers_count'] > 0:
            #     self.df_cleaned = self.df_cleaned[~outlier_mask].copy()

        cleaning_stats['after_outlier_removal'] = len(self.df_cleaned)  # This will be same as after_missing_occ

        # Calculate final statistics
        cleaning_stats['final_count'] = len(self.df_cleaned)
        cleaning_stats['total_removed'] = cleaning_stats['initial_count'] - cleaning_stats['final_count']
        cleaning_stats['removal_percentage'] = round(
            (cleaning_stats['total_removed'] / cleaning_stats['initial_count']) * 100, 2
        ) if cleaning_stats['initial_count'] > 0 else 0

        # Store cleaning statistics in data flow stats
        self.data_flow_stats['cleaning_pipeline'] = cleaning_stats

        if 'material_type' not in self.df_cleaned.columns:
            print("Warning: 'material_type' column not found. Filling with None.")
            self.df_cleaned['material_type'] = None

        if 'foundation_type' not in self.df_cleaned.columns:
            print("Warning: 'foundation_type' column not found. Filling with None.")
            self.df_cleaned['foundation_type'] = None

        print("  Applying new filter: Removing all rows with original HEIGHT <= 0...")
        h_numeric_raw = pd.to_numeric(self.df_cleaned['HEIGHT'], errors='coerce')
        invalid_h_mask = (h_numeric_raw.notna()) & (h_numeric_raw <= 0)
        invalid_h_count = int(invalid_h_mask.sum())

        if invalid_h_count > 0:
            self.df_cleaned = self.df_cleaned[~invalid_h_mask].copy()


        cleaning_stats['invalid_raw_height_count'] = invalid_h_count



        h = (self.df_cleaned['HEIGHT'].apply(pd.to_numeric, errors='coerce')
             if 'HEIGHT' in self.df_cleaned.columns
             else pd.Series(np.nan, index=self.df_cleaned.index))

        ph = (self.df_cleaned['PRED_HEIGHT'].apply(pd.to_numeric, errors='coerce')
              if 'PRED_HEIGHT' in self.df_cleaned.columns
              else pd.Series(np.nan, index=self.df_cleaned.index))


        self.df_cleaned['HEIGHT_USED'] = np.where(h.notna() & (h > 0), h, ph)
        self.df_cleaned['Assumed height'] = self.df_cleaned['HEIGHT_USED']


        count_before_height_filter = len(self.df_cleaned)

        invalid_assumed_height_mask = (self.df_cleaned['Assumed height'].isna()) | (
                    self.df_cleaned['Assumed height'] <= 0)
        num_invalid_assumed_heights = int(invalid_assumed_height_mask.sum())

        cleaning_stats['invalid_assumed_height_count'] = num_invalid_assumed_heights

        if num_invalid_assumed_heights > 0:
            self.df_cleaned = self.df_cleaned[~invalid_assumed_height_mask].copy()

        cleaning_stats['after_height_filter'] = len(self.df_cleaned)

        h_final = pd.to_numeric(self.df_cleaned['HEIGHT'], errors='coerce')


        mask_used_height = (h_final.notna())
        count_used_height = int(mask_used_height.sum())
        count_used_pred_height = len(self.df_cleaned) - count_used_height

        cleaning_stats['assumed_height_source'] = {
            'used_height': count_used_height,
            'used_pred_height': count_used_pred_height
        }




        cleaning_stats['final_count'] = len(self.df_cleaned)
        cleaning_stats['total_removed'] = cleaning_stats['initial_count'] - cleaning_stats['final_count']
        cleaning_stats['removal_percentage'] = round(
            (cleaning_stats['total_removed'] / cleaning_stats['initial_count']) * 100, 2
        ) if cleaning_stats['initial_count'] > 0 else 0

        # Store cleaning statistics in data flow stats
        self.data_flow_stats['cleaning_pipeline'] = cleaning_stats

        # Print summary of cleaning process
        print(f"Cleaned data: {len(self.df_cleaned)} records")
        print(f"  Removed {cleaning_stats['invalid_year_count']} records with invalid year")
        print(f"    - Invalid/zero: {cleaning_stats['invalid_year_details']['negative_or_zero']}")
        print(f"    - NaN values: {cleaning_stats['invalid_year_details']['nan_values']}")
        print(f"  Removed {cleaning_stats['missing_area_count']} records with missing area")


        print(
            f"  Removed {cleaning_stats.get('invalid_raw_height_count', 0)} records with original HEIGHT <= 0 (Step 1)")
        print(
            f"  Removed {cleaning_stats.get('invalid_assumed_height_count', 0)} records with invalid Assumed height (e.g., PRED_HEIGHT <= 0) (Step 2)")


        print(f"  Removed {cleaning_stats['area_outliers_count']} area outliers")
        if cleaning_stats['area_outlier_threshold']:
            print(f"    - Outlier threshold: {cleaning_stats['area_outlier_threshold']:,.2f} sqm")
        print(f"  Total removed: {cleaning_stats['total_removed']} ({cleaning_stats['removal_percentage']}%)")


        if 'assumed_height_source' in cleaning_stats:
            print(f"  Assumed height source (final data):")
            print(f"    - From HEIGHT: {cleaning_stats['assumed_height_source']['used_height']:,}")
            print(f"    - From PRED_HEIGHT: {cleaning_stats['assumed_height_source']['used_pred_height']:,}")


        # Store the cleaning stats for later use
        self.data_flow_stats['cleaning_stats'] = cleaning_stats

        return self

    def prepare_clustering_data(self, remove_outliers=False):
        """Prepare data for clustering"""
        print("Preparing clustering data...")

        # --- FINAL FIX: Include ALL columns needed by any function that uses df_cluster ---
        features = [
            'OCC_CLS', 'Est GFA sqmeters', 'SQMETERS', 'HEIGHT_USED',
            'PRED_HEIGHT', 'year_built', 'material_type', 'foundation_type'
        ]
        self.df_cluster = self.df_cleaned[features].dropna().copy()

        if remove_outliers:
            # Outlier removal should use GFA as it was originally
            area_threshold = self.df_cluster['Est GFA sqmeters'].quantile(0.99999)
            print(f"Area threshold for outliers: {area_threshold:,.2f} sqm")

            # Filter out the outliers
            initial_count = len(self.df_cluster)
            self.df_cluster = self.df_cluster[self.df_cluster['Est GFA sqmeters'] < area_threshold].copy()
            outliers_removed = initial_count - len(self.df_cluster)
            print(f"Records after removing outliers: {len(self.df_cluster)}")

            self.data_flow_stats['outliers_removed'] = outliers_removed

        return self

        # data_preprocessor.py

    def resolve_unclassified_from_occdict(self):
            """
            Reclassify rows whose OCC_CLS == 'Unclassified' (or NaN) using OCC_DICT vote counts.
            JSON-only, compact log:
              - Stores a compact mapping under data_flow_stats['unclassified_resolution']:
                * 'meta': {'id_field': <str>, 'class_legend': {'R':'Residential','C':...}}
                * 'unclassified_map': [[id, 'C'], [id2, 'R'], ...]  # <--- compact pairs
              - No CSV export, no extra DataFrame columns (to keep JSON small).
            Rules:
              - REL merges into COM (COM_eff = COM + REL).
              - If all considered classes sum to 0, keep Unclassified.
              - Tie-breaker priority: RES > COM > IND > AGR > GOV > EDU. # <-- Added AGR
            """
            import re
            from collections import Counter

            if not hasattr(self, 'df_cleaned'):
                print("resolve_unclassified_from_occdict: no df_cleaned; skip.")
                return
            for col in ('OCC_CLS', 'OCC_DICT'):
                if col not in self.df_cleaned.columns:
                    print("resolve_unclassified_from_occdict: missing OCC_CLS or OCC_DICT; skip.")
                    return

            # Preserve original label once for auditing (very small)
            if 'OCC_CLS_ORIG' not in self.df_cleaned.columns:
                self.df_cleaned['OCC_CLS_ORIG'] = self.df_cleaned['OCC_CLS']

            # --- MODIFICATION START: Include both 'Unclassified' string and NaN values ---
            # Create a mask for rows where OCC_CLS is 'Unclassified' OR NaN (null)
            uncls_mask = (self.df_cleaned['OCC_CLS'].astype(str).str.strip().str.lower() == 'unclassified') | \
                         (self.df_cleaned['OCC_CLS'].isna())
            # --- MODIFICATION END ---
            total_uncls_before = int(uncls_mask.sum())

            # Compact ID selection (prefer a stable id column; fallback to row index)
            candidate_id_cols = [c for c in ['BUILD_ID', 'UUID', 'OBJECTID', 'OGC_FID', 'fid', 'id'] if
                                 c in self.df_cleaned.columns]
            id_col = candidate_id_cols[0] if candidate_id_cols else None

            # Token regex
            pair_re = re.compile(r'([A-Z]{3})\s*:\s*(-?\d+(?:\.\d+)?)', flags=re.IGNORECASE)

            # --- MODIFICATION START: Added 'Agriculture'/'A' and 'Assembly'/'S' ---
            # Priority and legend (single-letter -> full name)
            priority = ['Residential', 'Commercial', 'Industrial', 'Agriculture', 'Government',
                        'Education', 'Assembly']  # Added Agriculture and Assembly
            to_code = {'Residential': 'R', 'Commercial': 'C', 'Industrial': 'I', 'Agriculture': 'A', 'Government': 'G',
                       'Education': 'E', 'Assembly': 'S'}  # Added A, S
            legend = {'R': 'Residential', 'C': 'Commercial', 'I': 'Industrial', 'A': 'Agriculture', 'G': 'Government',
                      'E': 'Education', 'S': 'Assembly'}  # Added A, S

            # --- MODIFICATION END ---

            # Helpers
            def _has_text(x):
                if x is None:
                    return False
                s = str(x).strip()
                return bool(s) and s.lower() != 'nan'

            # Accumulators
            changed_to = Counter()
            changed = 0
            unchanged_zero_or_unparsable = 0
            tie_situations = Counter()  # <-- ADD THIS LINE to track tie reasons
            # --- COMPACT MAPPING (THIS IS WHAT GOES TO JSON) ---
            # Each item: [id_value_or_index, 'R'|'C'|'I'|'A'|'G'|'E']
            reclass_pairs = []

            # Get the indices where the mask is True
            idxs = self.df_cleaned.index[uncls_mask]

            # Iterate only over the selected indices
            for i in idxs:
                occ_txt = self.df_cleaned.at[i, 'OCC_DICT']
                if not _has_text(occ_txt):
                    unchanged_zero_or_unparsable += 1
                    continue

                # Parse KEY:val tokens
                pairs = {}
                for k, v in pair_re.findall(str(occ_txt)):
                    try:
                        # Sum up values if a key appears multiple times (case-insensitive)
                        pairs[k.upper()] = pairs.get(k.upper(), 0) + int(float(v))
                    except Exception:
                        pass  # Ignore parsing errors for a value
                if not pairs:
                    unchanged_zero_or_unparsable += 1
                    continue

                # --- MODIFICATION START: Extract AGR count ---
                res = int(pairs.get('RES', 0))
                com = int(pairs.get('COM', 0))
                ind = int(pairs.get('IND', 0))
                gov = int(pairs.get('GOV', 0))
                edu = int(pairs.get('EDU', 0))
                rel = int(pairs.get('REL', 0))
                agr = int(pairs.get('AGR', 0))  # Extract AGR count
                # --- MODIFICATION END ---

                # Calculate total votes across considered categories
                total_votes = res + com + ind + gov + edu + agr + rel

                if total_votes == 0:
                    unchanged_zero_or_unparsable += 1
                    continue

                # Create a dictionary of scores for comparison
                scores = {
                    'Residential': res,
                    'Commercial': com,
                    'Industrial': ind,
                    'Agriculture': agr,
                    'Government': gov,
                    'Education': edu,
                    'Assembly': rel
                }
                # --- MODIFICATION END ---

                # Find the maximum score
                mx = max(scores.values())

                # Find all categories that achieved the maximum score
                winners = [k for k, v in scores.items() if v == mx]

                # Determine the chosen category

                # Determine the chosen category
                if len(winners) > 1:
                    # NEW RULE: If there is a tie for the max score (e.g., RES:1, COM:1),
                    # do not reclassify. Keep it as 'Unclassified'.

                    # --- START: Log the tie situation ---
                    try:
                        # Get the abbreviations for the winners (e.g., 'R', 'C')
                        sorted_tied_codes = sorted([to_code[w] for w in winners])
                        # Create a key string, e.g., "C:1, R:1" (using int(mx) for clean key)
                        tie_key = ", ".join([f"{code}:{int(mx)}" for code in sorted_tied_codes])
                        # Increment the counter for this specific tie combination
                        tie_situations[tie_key] += 1
                    except Exception as e:
                        # Fail safe in case of unseen errors, just don't log this tie
                        print(f"Warning: Failed to log tie situation - {e}")
                    # --- END: Log the tie situation ---

                    unchanged_zero_or_unparsable += 1
                    continue  # Skip to the next building

                # If we are here, len(winners) == 1, meaning there is a single, clear winner.
                chosen = winners[0]
                # Apply the new label to the DataFrame column 'OCC_CLS'
                self.df_cleaned.at[i, 'OCC_CLS'] = chosen

                # Record the reclassification in a compact format [id, chosen_code]
                rec_id = self.df_cleaned.at[i, id_col] if id_col else i  # Use ID column if available, else index
                reclass_pairs.append([rec_id, to_code[chosen]])  # Use the code mapping (e.g., 'A' for Agriculture)

                # Increment counters for statistics
                changed += 1
                changed_to[chosen] += 1  # Track counts for each resulting category

            # Sort ties by frequency, descending, for cleaner JSON output
            sorted_ties = dict(tie_situations.most_common())

            stats_payload = {
                'meta': {
                    'id_field': id_col if id_col else 'row_index',
                    'class_legend': legend  # Legend now includes 'A': 'Agriculture'
                },
                'total_unclassified_before': int(total_uncls_before),
                'with_occdict': int(self.df_cleaned.loc[uncls_mask, 'OCC_DICT'].apply(_has_text).sum()),
                'changed': int(changed),
                'unchanged_zero_or_unparsable': int(unchanged_zero_or_unparsable),
                'changed_to_counts': {k: int(v) for k, v in changed_to.items()},  # Counts now include Agriculture

                # --- ADD THIS LINE ---
                'tie_situations_logged': {k: int(v) for k, v in sorted_ties.items()},
                # --- END OF ADDED LINE ---

                # The full mapping of each reclassified Unclassified row (compact!)
                'unclassified_map': reclass_pairs
            }

            # Store the statistics payload in the class attribute for later export
            if not hasattr(self, 'data_flow_stats'):
                self.data_flow_stats = {}
            self.data_flow_stats['unclassified_resolution'] = stats_payload

            # Print a summary to the console during script execution
            print("Unclassified reclassification (JSON-only) summary:",
                  {k: stats_payload[k] for k in [
                      'total_unclassified_before', 'with_occdict', 'changed',
                      'unchanged_zero_or_unparsable'
                  ]})
            print(f"  Breakdown of reclassified types: {dict(changed_to)}")

    def recalculate_mix_sc_for_reclassified(self):
        """
        Recalculate MIX_SC for rows that were just reclassified from 'Unclassified'.
        This must be run *after* resolve_unclassified_from_occdict.
        """
        print("Recalculating MIX_SC for reclassified 'Unclassified' buildings...")

        # 1. Find all rows that were reclassified from 'Unclassified'
        if 'OCC_CLS_ORIG' not in self.df_cleaned.columns:
            print("  Warning: 'OCC_CLS_ORIG' column not found. Skipping MIX_SC recalculation.")
            return self

        reclassified_mask = (
                (self.df_cleaned['OCC_CLS_ORIG'] == 'Unclassified') &
                (self.df_cleaned['OCC_CLS'] != 'Unclassified')
        )
        reclassified_indices = self.df_cleaned.index[reclassified_mask]

        if len(reclassified_indices) == 0:
            print("  No buildings were reclassified. Skipping.")
            return self

        # 2. Mapping from new 'OCC_CLS' to NSI point types
        CLS_TO_NSI_TYPES = {
            'Residential': ['RES'],
            'Commercial': ['COM'],
            'Industrial': ['IND'],
            'Government': ['GOV'],
            'Education': ['EDU'],
            'Agriculture': ['AGR'],
            'Assembly': ['REL']
        }

        # 3. Regex for parsing 'OCC_DICT' strings
        pair_re = re.compile(r'([A-Z]{3})\s*:\s*(-?\d+(?:\.\d+)?)', flags=re.IGNORECASE)

        # 4. Recalculate MIX_SC for each reclassified row
        recalculated_count = 0
        for i in reclassified_indices:
            new_cls = self.df_cleaned.at[i, 'OCC_CLS']
            occ_dict_str = self.df_cleaned.at[i, 'OCC_DICT']

            same_types = CLS_TO_NSI_TYPES.get(new_cls, [])

            occ_counts = {}
            if pd.notna(occ_dict_str):
                for k, v in pair_re.findall(str(occ_dict_str)):
                    try:
                        val_int = int(float(v))
                        if val_int > 0:
                            occ_counts[k.upper()] = occ_counts.get(k.upper(), 0) + val_int
                    except Exception:
                        pass

            # 5. Separate same-type and conflict-type points
            same_type_points = 0
            conflict_counts = {}

            for key, value in occ_counts.items():
                if key in same_types:
                    same_type_points += value
                else:
                    conflict_counts[key] = value

            # 6. Apply MIX_SC rules
            total_same_points = same_type_points
            total_conflict_types = len(conflict_counts)

            new_mix_sc = self.df_cleaned.at[i, 'MIX_SC']

            # Rule 1: Same Type Only (NaN)
            if total_same_points > 0 and total_conflict_types == 0:
                new_mix_sc = np.nan

            # Rule 2: 1 Conflict Type (MIX_SC1)
            elif total_same_points == 0 and total_conflict_types == 1:
                new_mix_sc = 'MIX_SC1'

            # Rule 3: Same & Different Types (MIX_SC2)
            elif total_same_points > 0 and total_conflict_types > 0:
                new_mix_sc = 'MIX_SC2'

            # Rule 4: >1 Conflict Types (MIX_SC3)
            elif total_same_points == 0 and total_conflict_types > 1:
                new_mix_sc = 'MIX_SC3'

            # Rule 5: If occ_counts is empty, no update needed.

            # 7. Update DataFrame
            self.df_cleaned.at[i, 'MIX_SC'] = new_mix_sc
            recalculated_count += 1

        print(f"  Recalculated MIX_SC for {recalculated_count} buildings.")

        if 'unclassified_resolution' in self.data_flow_stats:
            self.data_flow_stats['unclassified_resolution']['mix_sc_recalculated_count'] = recalculated_count

        return self

    def process_hierarchical_distribution(self):
        """
        Processes hierarchical data for Sankey diagrams for multiple views.
        - by_count: Standard hierarchy with values as building counts.
        - by_gfa: Standard hierarchy with values as summed GFA.
        - by_count_simplified: Simplified hierarchy (no area/height) with values as counts.
        """
        print("Processing hierarchical distribution for multiple views...")

        df_work = self.df_cleaned.copy()
        if 'drainagecl' in df_work.columns:
            df_work['drainage_cat'] = df_work['drainagecl'].fillna('Unknown Drainage')
        else:
            df_work['drainage_cat'] = 'Unknown Drainage'
        df_work['drainage_cat'] = df_work['drainage_cat'].astype('category')

        # Define consistent, global bins
        area_percentiles = df_work['Est GFA sqmeters'].quantile([0.33, 0.67]).values
        area_bins = [0, area_percentiles[0], area_percentiles[1], float('inf')]
        area_labels = ['Small', 'Medium', 'Large']

        height_percentiles = df_work['HEIGHT_USED'].dropna().quantile([0.33, 0.67]).values
        height_bins = [0, height_percentiles[0], height_percentiles[1], float('inf')]
        height_labels = ['Short', 'Mid', 'High']  # Renamed "Low" to "Short"

        year_bins = [0, 1940, 1980, float('inf')]
        year_labels = ['Historic (<1940)', 'Mid-Century (40-80)', 'Modern (>1980)']

        # Apply bins
        df_work['area_cat'] = pd.cut(df_work['Est GFA sqmeters'], bins=area_bins, labels=area_labels, right=False)
        df_work['height_cat'] = pd.cut(df_work['HEIGHT_USED'], bins=height_bins, labels=height_labels, right=False)
        df_work['year_cat'] = pd.cut(df_work['year_built'], bins=year_bins, labels=year_labels, right=False)

        # Main dictionary to hold all versions
        hierarchical_data = {}

        # Process for 'all' buildings view
        hierarchical_data['all'] = {
            'by_count': self._process_hierarchy(df_work,
                                                ['occ_cat', 'area_cat', 'height_cat', 'year_cat', 'drainage_cat'],
                                                'count'),
            'by_gfa': self._process_hierarchy(df_work,
                                              ['occ_cat', 'area_cat', 'height_cat', 'year_cat', 'drainage_cat'], 'gfa'),
            'by_count_simplified': self._process_hierarchy(df_work, ['occ_cat', 'year_cat', 'drainage_cat'], 'count'),
            'by_gfa_simplified': self._process_hierarchy(df_work, ['occ_cat', 'year_cat', 'drainage_cat'], 'gfa')
        }
        # Add binning info for the UI
        bin_info = {
            'Area': f"Small (<{area_bins[1]:.0f} sqm), Medium ({area_bins[1]:.0f}-{area_bins[2]:.0f} sqm), Large (>{area_bins[2]:.0f} sqm)",
            'Height': f"Short (<{height_bins[1]:.1f}m), Mid ({height_bins[1]:.1f}-{height_bins[2]:.1f}m), High (>{height_bins[2]:.1f}m)",
            'Year': f"Historic (<{year_bins[1]}), Mid-Century ({year_bins[1]}-{year_bins[2]}), Modern (>{year_bins[2]})",
            'Drainage': "Multiple classes including Well, Moderately, Poorly drained, etc."
        }
        for view in hierarchical_data['all']:
            hierarchical_data['all'][view]['bin_info'] = bin_info

        # Process for each individual occupancy class
        for occ_class in df_work['OCC_CLS'].unique():
            occ_data = df_work[df_work['OCC_CLS'] == occ_class]
            if len(occ_data) > 100:
                hierarchical_data[occ_class] = {
                    'by_count': self._process_hierarchy(occ_data,
                                                        ['area_cat', 'height_cat', 'year_cat', 'drainage_cat'], 'count',
                                                        root_name=occ_class),
                    'by_gfa': self._process_hierarchy(occ_data, ['area_cat', 'height_cat', 'year_cat', 'drainage_cat'],
                                                      'gfa', root_name=occ_class),
                    'by_count_simplified': self._process_hierarchy(occ_data, ['year_cat', 'drainage_cat'], 'count',
                                                                   root_name=occ_class),
                    'by_gfa_simplified': self._process_hierarchy(occ_data, ['year_cat', 'drainage_cat'], 'gfa',
                                                                 root_name=occ_class)
                }
                for view in hierarchical_data[occ_class]:
                    hierarchical_data[occ_class][view]['bin_info'] = bin_info

        print(f"  Processed hierarchical data for {len(hierarchical_data)} occupancy classes across 3 views")
        return hierarchical_data

    # NEW: Full-population aggregation for Year → Occupancy → Material → Foundation → Soil (compname)
    def process_year_occ_mat_found_soil_flow(self, top_n_occ=12, top_n_soils=15, group_others=False):
        """
        Build an aggregated flow table across the *full* cleaned dataset (no sampling).
        Output is compact (combination counts), ideal for front-end Sankey with toggles.
        - Year bands: Historic (<1940), Mid-Century (1940–1980), Modern (>1980)  (top→bottom order)
        - Occupancy: Top N by count + 'Other' (if group_others is True)
        - Material/Foundation: use string values or 'Unknown'
        - Soil: compname Top N + 'Other Soils' (if group_others is True)
        """
        df = self.df_cleaned.copy()

        # --- Year band (consistent labels for the front-end order override) ---
        year_bins = [0, 1940, 1980, float('inf')]
        year_labels = ['Historic (<1940)', 'Mid-Century (1940–1980)', 'Modern (>1980)']
        df['year_band'] = pd.cut(df['year_built'], bins=year_bins, labels=year_labels, right=False)

        # --- Occupancy (top-N + Other) ---
        occ_col = 'OCC_CLS'
        if group_others:
            occ_counts = df[occ_col].value_counts(dropna=False)
            top_occ = occ_counts.nlargest(top_n_occ).index
            df['occupancy'] = df[occ_col].where(df[occ_col].isin(top_occ), other='Other')
        else:
            df['occupancy'] = df[occ_col].fillna('Unknown')

        # --- Material / Foundation / Soil(compname) ---
        df['material'] = df['material_type'].fillna('Unknown') if 'material_type' in df.columns else 'Unknown'
        df['foundation'] = df['foundation_type'].fillna('Unknown') if 'foundation_type' in df.columns else 'Unknown'
        df['soil'] = df['compname'].fillna('Unknown') if 'compname' in df.columns else 'Unknown'

        # Soil: top-N + Other Soils
        if group_others:
            soil_counts = df['soil'].value_counts(dropna=False)
            top_soils = soil_counts.nlargest(top_n_soils).index
            df['soil'] = df['soil'].where(df['soil'].isin(top_soils), other='Other Soils')

        # --- Group by the full chain (count + GFA) ---
        group_cols = ['year_band', 'occupancy', 'material', 'foundation', 'soil']

        # Count buildings per combination
        grp_count = (
            df.groupby(group_cols, observed=True)
            .size()
            .reset_index(name='count')
        )

        # Sum GFA per combination
        grp_gfa = (
            df.groupby(group_cols, observed=True)['Est GFA sqmeters']
            .sum()
            .reset_index(name='gfa')
        )

        # Merge both metrics into one table
        grp = grp_count.merge(grp_gfa, on=group_cols, how='left')
        grp['gfa'] = grp['gfa'].fillna(0.0).astype(float)  # ensure numeric

        # Convert to list-of-dicts for compact JSON
        combination_counts = grp.to_dict(orient='records')

        return {
            'combination_counts': combination_counts,
            'meta': {
                'total_buildings': int(len(df)),
                'levels': ['All Buildings', 'Year', 'Occupancy', 'Material', 'Foundation', 'Soil'],
                'year_order_top_to_bottom': ['Historic (<1940)', 'Mid-Century (1940–1980)', 'Modern (>1980)'],
                'available_metrics': ['count', 'gfa'],  # values: count = buildings, gfa = sqm
                'gfa_units': 'sqm',
                'grouping': {
                    'occupancy': f'top_{top_n_occ}_plus_other' if group_others else 'raw',
                    'soil': f'top_{top_n_soils}_plus_other' if group_others else 'raw'
                }
            }
        }

    def _process_hierarchy(self, df, levels, value_mode='count', root_name='All Buildings'):
        """
        A generic function to process hierarchical data for Sankey diagrams.
        Can generate diagrams based on count or GFA, and with different levels.
        """
        sankey_data = {'nodes': [], 'links': []}

        df_proc = df.copy()

        # For the 'all' view, create the top-level occupancy category
        if 'occ_cat' in levels:
            occ_counts = df_proc['OCC_CLS'].value_counts()
            top_9_occ = occ_counts.nlargest(9).index.tolist()
            df_proc['occ_cat'] = df_proc['OCC_CLS'].apply(lambda x: x if x in top_9_occ else 'Other')

        # Define the full hierarchy including the root
        full_hierarchy = [root_name] + levels

        # Group data and create links
        for i in range(len(full_hierarchy) - 1):
            source_level = full_hierarchy[i]
            target_level = full_hierarchy[i + 1]

            group_by_cols = [source_level, target_level] if i > 0 else [target_level]

            if value_mode == 'gfa':
                # Aggregate by summing 'Est GFA sqmeters' if value_mode is 'gfa'
                agg_result = df_proc.groupby(group_by_cols, observed=True).agg(
                    {'Est GFA sqmeters': 'sum'}).reset_index()
                agg_result.rename(columns={'Est GFA sqmeters': 'value'}, inplace=True)
            else:
                # Aggregate by counting rows if value_mode is 'count'
                agg_result = df_proc.groupby(group_by_cols, observed=True).size().reset_index(name='value')

            for _, row in agg_result.iterrows():
                source_name = row.get(source_level, root_name)
                sankey_data['links'].append({
                    'source': str(source_name),
                    'target': str(row[target_level]),
                    'value': row['value']
                })

        # Collect all unique nodes from the links
        node_names = set()
        for link in sankey_data['links']:
            node_names.add(link['source'])
            node_names.add(link['target'])

        # Create the node list and assign a level (for coloring)
        node_map = {name: {'name': name} for name in node_names}
        for i, level_name in enumerate(full_hierarchy):
            if level_name in df_proc.columns:
                for category in df_proc[level_name].unique():
                    if str(category) in node_map:
                        node_map[str(category)]['level'] = i
        if root_name in node_map:
            node_map[root_name]['level'] = 0

        sankey_data['nodes'] = list(node_map.values())
        sankey_data['total_buildings'] = len(df)

        return sankey_data

    def process_occupancy_hierarchy(self):
        """Processes the hierarchy from OCC_CLS to PRIM_OCC for a Sankey diagram, showing ALL categories."""
        print("Processing OCC_CLS to PRIM_OCC hierarchy (showing all categories)...")

        df = self.df_cleaned[['OCC_CLS', 'PRIM_OCC']].dropna()
        all_links = df.groupby(['OCC_CLS', 'PRIM_OCC']).size().reset_index(name='value')

        final_links = all_links[all_links['value'] > 0].copy()


        final_links['OCC_CLS_mod'] = final_links['OCC_CLS'].astype(str) + ' (Class)'


        condition = final_links['PRIM_OCC'] == 'Unclassified'
        true_values = 'Unclassified (from ' + final_links['OCC_CLS'] + ')'
        false_values = final_links['PRIM_OCC'].astype(str) + ' (Type)'

        final_links['PRIM_OCC_mod'] = np.where(condition, true_values, false_values)

        occ_cls_nodes = final_links['OCC_CLS_mod'].unique().tolist()
        prim_occ_nodes = final_links['PRIM_OCC_mod'].unique().tolist()

        all_node_labels = occ_cls_nodes + prim_occ_nodes
        node_map = {name: i for i, name in enumerate(all_node_labels)}

        sankey_nodes = [{'name': name} for name in all_node_labels]
        sankey_links = {
            'source': final_links['OCC_CLS_mod'].map(node_map).tolist(),
            'target': final_links['PRIM_OCC_mod'].map(node_map).tolist(),
            'value': final_links['value'].tolist()
        }

        return {
            'nodes': sankey_nodes,
            'links': sankey_links
        }

    def process_occ_cls_to_occdict_sankey(self):
        """
        Build Sankey: OCC_CLS (left) → NSI occtype (right), using the OCC_DICT column.
        Exposes two metrics:
          - by_points:   Sum of NSI points per occtype（∑点数）
          - by_buildings:Count of buildings having ≥1 point of that occtype（≥1即记1）
        """
        print("Processing OCC_CLS → NSI occtype (OCC_DICT) sankey...")

        import pandas as pd
        df = self.df_cleaned[['OCC_CLS', 'OCC_DICT']].copy()

        # Robust parser: accepts dict or string like "RES: 1, COM: 0, ..."
        def parse_occ_dict(v):
            if isinstance(v, dict):
                return v
            if pd.isna(v):
                return {}
            s = str(v).strip().strip('{}')
            parts = [p for p in s.replace(';', ',').split(',') if p.strip()]
            out = {}
            for p in parts:
                if ':' in p:
                    k, val = p.split(':', 1)
                    k = k.strip()
                    try:
                        val = int(float(val.strip()))
                    except Exception:
                        val = 0
                    out[k] = val
            return out

        rows = []
        for _, r in df.iterrows():
            occ = r['OCC_CLS']
            d = parse_occ_dict(r['OCC_DICT'])
            for k, v in d.items():
                rows.append({
                    'OCC_CLS': str(occ),
                    'occtype': str(k),
                    'points': int(v),
                    'has': 1 if int(v) > 0 else 0
                })

        if not rows:
            return None

        x = pd.DataFrame(rows)


        agg_points = (
            x.groupby(['OCC_CLS', 'occtype'], observed=True)['points']
            .sum()
            .reset_index()
        )
        agg_points = agg_points[agg_points['points'] > 0].copy()


        agg_buildings = (
            x[x['has'] > 0]
            .groupby(['OCC_CLS', 'occtype'], observed=True)
            .size()
            .reset_index(name='buildings')
        )

        def to_sankey(agg_df, value_col):
            left = agg_df['OCC_CLS'].astype(str) + ' (Class)'
            # MODIFICATION START: Make target nodes unique by appending the source class
            right = agg_df['occtype'].astype(str) + ' (from ' + agg_df['OCC_CLS'] + ')'
            # MODIFICATION END
            nodes = pd.Index(pd.concat([left, right], ignore_index=True).unique())
            idx = {name: i for i, name in enumerate(nodes)}
            return {
                'nodes': [{'name': n} for n in nodes],
                'links': {
                    'source': [idx[s] for s in left],
                    'target': [idx[t] for t in right],
                    'value': agg_df[value_col].astype(float).tolist()
                }
            }

        return {
            'by_points': to_sankey(agg_points.rename(columns={'points': 'value'}), 'value'),
            'by_buildings': to_sankey(agg_buildings.rename(columns={'buildings': 'value'}), 'value')
        }

    def perform_clustering(self, n_clusters=7):
        """Perform K-means clustering"""
        print(f"Performing K-means clustering with {n_clusters} clusters...")

        # --- FIX: Updated the numerical features to SQMETERS and PRED_HEIGHT ---
        # The ColumnTransformer now scales the correct columns for the model.
        numerical_features_for_clustering = ['SQMETERS', 'HEIGHT_USED', 'year_built']

        # Set up preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features_for_clustering),
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['OCC_CLS'])
            ])

        # Define the full feature set for transformation
        features_for_transform = numerical_features_for_clustering + ['OCC_CLS']

        # Transform data using the correct feature set
        X_prepared = self.preprocessor.fit_transform(self.df_cluster[features_for_transform])

        # Run K-means
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        self.df_cluster['cluster'] = self.kmeans.fit_predict(X_prepared)

        print("Clustering complete")
        return self

    def calculate_elbow_scores(self, k_range=range(2, 16)):
        """Calculate WCSS scores for elbow method"""
        print("Calculating elbow scores...")

        features = ['OCC_CLS', 'Est GFA sqmeters', 'year_built']
        df_temp = self.df_cleaned[features].dropna()

        # Remove outliers
        area_threshold = df_temp['Est GFA sqmeters'].quantile(0.99999)
        df_temp = df_temp[df_temp['Est GFA sqmeters'] < area_threshold].copy()

        # Preprocess
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['Est GFA sqmeters', 'year_built']),
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['OCC_CLS'])
            ])
        X_prepared = preprocessor.fit_transform(df_temp)

        # Calculate WCSS
        wcss = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(X_prepared)
            wcss.append(kmeans.inertia_)
            print(f"  Computed k={k}")

        return list(k_range), wcss

    def _get_cluster_assignments_for_df(self, df_subset, feature_combo, k):
        """
        Perform clustering and return cluster assignments for each row in the dataframe.
        feature_combo can be: 'base', 'material', 'foundation', 'both'
        """
        if len(df_subset) < k:
            return None

        # Prepare features based on combination
        numerical_features = ['SQMETERS', 'HEIGHT_USED', 'year_built']
        categorical_features = ['OCC_CLS']

        if feature_combo == 'material' or feature_combo == 'both':
            if 'material_type' in df_subset.columns and df_subset['material_type'].notna().any():
                categorical_features.append('material_type')
        if feature_combo == 'foundation' or feature_combo == 'both':
            if 'foundation_type' in df_subset.columns and df_subset['foundation_type'].notna().any():
                categorical_features.append('foundation_type')

        # Check if all features exist
        all_features = numerical_features + categorical_features
        for feat in all_features:
            if feat not in df_subset.columns:
                print(f"    Warning: Feature '{feat}' not found for clustering. Skipping.")
                return None

        # Drop rows with NaN in any of the selected features for this specific clustering run
        df_clusterable = df_subset[all_features].dropna()
        if len(df_clusterable) < k:
            return None

        # Setup preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ], remainder='passthrough')

        try:
            # Transform and cluster
            X_prepared = preprocessor.fit_transform(df_clusterable[all_features])
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(X_prepared)

            # Create a Series with the original index to map results back
            cluster_series = pd.Series(clusters, index=df_clusterable.index)

            # Return a series that can be aligned back to the original df_subset
            return cluster_series

        except Exception as e:
            print(f"    Error in clustering with {feature_combo} for assignments: {e}")
            return None

    def _perform_clustering_with_features(self, df_subset, feature_combo, k):
        """
        Perform clustering with specific feature combination
        feature_combo can be: 'base', 'material', 'foundation', 'both'
        """
        if len(df_subset) < k:
            return None

        # MODIFICATION: Changed numerical features to use footprint area and height instead of GFA.
        # PREVIOUSLY: numerical_features = ['Est GFA sqmeters', 'year_built']
        numerical_features = ['SQMETERS', 'HEIGHT_USED', 'year_built']
        categorical_features = ['OCC_CLS']

        if feature_combo == 'material' or feature_combo == 'both':
            categorical_features.append('material_type')
        if feature_combo == 'foundation' or feature_combo == 'both':
            categorical_features.append('foundation_type')

        # Check if all features exist
        all_features = numerical_features + categorical_features
        for feat in all_features:
            if feat not in df_subset.columns:
                return None

        # Setup preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        try:
            # Transform and cluster
            X_prepared = preprocessor.fit_transform(df_subset[all_features])
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(X_prepared)

            # Calculate statistics
            cluster_stats = []
            for cluster_id in range(k):
                cluster_mask = clusters == cluster_id
                cluster_data = df_subset[cluster_mask]

                if len(cluster_data) == 0:
                    continue

                stats = {
                    'cluster_id': cluster_id,
                    'count': len(cluster_data),
                    'avg_area': float(cluster_data['SQMETERS'].mean()),
                    'std_area': float(cluster_data['SQMETERS'].std(ddof=0)),
                    'avg_sqmeters': float(cluster_data['SQMETERS'].mean()),
                    'std_sqmeters': float(cluster_data['SQMETERS'].std(ddof=0)),
                    'avg_height': float(cluster_data['HEIGHT_USED'].mean()),
                    'std_height': float(cluster_data['HEIGHT_USED'].std(ddof=0)),
                    'avg_year': int(cluster_data['year_built'].mean()),
                    'std_year': float(cluster_data['year_built'].std(ddof=0))
                }

                # Add dominant material/foundation if applicable
                if 'material_type' in categorical_features:
                    material_counts = cluster_data['material_type'].value_counts()
                    if len(material_counts) > 0:
                        stats['dominant_material'] = material_counts.index[0]

                if 'foundation_type' in categorical_features:
                    foundation_counts = cluster_data['foundation_type'].value_counts()
                    if len(foundation_counts) > 0:
                        stats['dominant_foundation'] = foundation_counts.index[0]

                cluster_stats.append(stats)

            return {
                'wcss': float(kmeans.inertia_),
                'clusters': cluster_stats
            }
        except Exception as e:
            print(f"    Error in clustering with {feature_combo}: {e}")
            return None

    def _get_cluster_stats_for_df(self, df_to_cluster):
        """Helper function to perform clustering and get stats for a given dataframe"""
        k_results = {}

        if len(df_to_cluster) < 10:
            return None

        # MODIFICATION: Changed features for scaling to use SQMETERS and PRED_HEIGHT.
        # PREVIOUSLY: X_scaled = scaler.fit_transform(df_to_cluster[['Est GFA sqmeters', 'year_built']])
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_to_cluster[['SQMETERS', 'HEIGHT_USED', 'year_built']])

        # Perform clustering for different k values (2-7)
        for k in range(2, 10):
            if len(df_to_cluster) < k:
                continue

            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(X_scaled)
            df_to_cluster[f'cluster_k{k}'] = clusters

            # Analyze clusters
            cluster_stats = []
            for cluster_id in range(k):
                cluster_data = df_to_cluster[df_to_cluster[f'cluster_k{k}'] == cluster_id]

                if len(cluster_data) == 0: continue

                # MODIFICATION: Changed stats to reflect new dimensions.
                # PREVIOUSLY: 'avg_area': float(cluster_data['Est GFA sqmeters'].mean()), 'std_area': float(cluster_data['Est GFA sqmeters'].std(ddof=0))
                cluster_stats.append({
                    'cluster_id': cluster_id,
                    'count': len(cluster_data),
                    'avg_area': float(cluster_data['SQMETERS'].mean()),
                    'std_area': float(cluster_data['SQMETERS'].std(ddof=0)),
                    'avg_sqmeters': float(cluster_data['SQMETERS'].mean()),
                    'std_sqmeters': float(cluster_data['SQMETERS'].std(ddof=0)),
                    'avg_gfa': float(cluster_data['Est GFA sqmeters'].mean()),
                    'std_gfa': float(cluster_data['Est GFA sqmeters'].std(ddof=0)),
                    'avg_height': float(cluster_data['HEIGHT_USED'].mean()),
                    'std_height': float(cluster_data['HEIGHT_USED'].std(ddof=0)),
                    'avg_year': int(cluster_data['year_built'].mean()),
                    'std_year': float(cluster_data['year_built'].std(ddof=0))
                })

            k_results[k] = {
                'wcss': float(kmeans.inertia_),
                'clusters': cluster_stats
            }
        return k_results

    def get_overview_occupancy_counts(self):
        """Get overall occupancy counts for all buildings (not just pre-1940)"""
        print("Calculating overview occupancy counts...")

        # Use all cleaned data
        occ_counts = self.df_cleaned['OCC_CLS'].value_counts()

        return occ_counts.to_dict()

    def process_mix_sc_distribution(self):
        """Calculates and formats the distribution of the MIX_SC column."""
        print("Processing MIX_SC distribution...")
        if 'MIX_SC' not in self.df_cleaned.columns:
            print("  Warning: 'MIX_SC' column not found. Skipping.")
            return None

        # Use value_counts with dropna=False to include NaN values
        counts = self.df_cleaned['MIX_SC'].value_counts(dropna=False)

        # Map the raw values to the descriptive labels you provided
        # The key for NaN from value_counts is actually the float `nan`
        mix_sc_data = {
            'Same Type Only': counts[counts.index.isna()].sum(),  # Sum up potential NaN values
            '1 Conflict Type (MIX_SC1)': counts.get('MIX_SC1', 0),
            'Same & Different Types (MIX_SC2)': counts.get('MIX_SC2', 0),
            '>1 Conflict Types (MIX_SC3)': counts.get('MIX_SC3', 0)
        }

        # Filter out any labels that might have a zero count, just in case
        return {k: int(v) for k, v in mix_sc_data.items() if v > 0}

    def process_temporal_data(self):
        """Process data for temporal analysis"""
        print("Processing temporal data...")

        temporal_data = []

        # Process by year
        for year in self.df_cluster['year_built'].unique():
            year_data = self.df_cluster[self.df_cluster['year_built'] == year]

            for occ_cls in year_data['OCC_CLS'].unique():
                occ_data = year_data[year_data['OCC_CLS'] == occ_cls]

                temporal_data.append({
                    'year': int(year),
                    'display_year': 'pre-1940' if int(year) < 1940 else str(int(year)),
                    'occupancy': occ_cls,
                    'count': len(occ_data),
                    'avg_area': float(occ_data['Est GFA sqmeters'].mean()),
                    'total_area': float(occ_data['Est GFA sqmeters'].sum())
                })

        return temporal_data

    def process_pre1940_data(self):
        """Process pre-1940 building data"""
        print("Processing pre-1940 data...")

        df_pre_1940 = self.df_cleaned[self.df_cleaned['year_built'] < 1940].copy()

        # Get occupancy counts
        occ_counts = df_pre_1940['OCC_CLS'].value_counts()

        pre1940_data = {
            'total_count': len(df_pre_1940),
            'occupancy_counts': occ_counts.to_dict(),
            'residential_count': int(occ_counts.get('Residential', 0)),
            'non_residential_count': int(occ_counts.drop('Residential', errors='ignore').sum()),
            'percentage_of_total': round(len(df_pre_1940) / len(self.df_cleaned) * 100, 2)
        }

        return pre1940_data

    def process_post1940_data(self):
        """Process post-1940 building data"""
        print("Processing post-1940 data...")

        df_post_1940 = self.df_cleaned[self.df_cleaned['year_built'] >= 1940].copy()

        # Process by decade
        decade_data = {}
        for decade in range(1940, 2030, 10):
            decade_df = df_post_1940[
                (df_post_1940['year_built'] >= decade) &
                (df_post_1940['year_built'] < decade + 10)
            ]

            if len(decade_df) > 0:
                decade_counts = decade_df['OCC_CLS'].value_counts()
                decade_data[f"{decade}s"] = {
                    'total': len(decade_df),
                    'occupancy_counts': decade_counts.to_dict()
                }

        return decade_data

    def process_occupancy_clusters_enhanced(self):
        """
        Process clustering for each occupancy class with multiple k values
        AND different feature combinations (base, +material, +foundation, +both)
        """
        print("Processing enhanced occupancy-specific clusters with feature combinations...")
        occupancy_clusters = {}

        # Feature combinations to test
        feature_combos = ['base', 'material', 'foundation', 'both']

        # First, process for "all" classes
        print("  Processing 'all' with multiple feature combinations...")
        # NEW and CORRECT
        features_extended = ['SQMETERS', 'HEIGHT_USED', 'year_built', 'OCC_CLS', 'material_type', 'foundation_type']

        df_all = self.df_cleaned[features_extended].dropna().copy()

        if len(df_all) > 10:
            all_results = {
                'total_buildings': len(df_all),
                'feature_combinations': {}
            }

            for combo in feature_combos:
                print(f"    Computing clustering for feature combo: {combo}")
                combo_results = {}

                for k in range(2, 8):
                    result = self._perform_clustering_with_features(df_all, combo, k)
                    if result:
                        combo_results[k] = result

                if combo_results:
                    all_results['feature_combinations'][combo] = combo_results

            occupancy_clusters['all'] = all_results

        # Then, process for each individual occupancy class
        for occ_class in self.df_cleaned['OCC_CLS'].unique():
            print(f"  Processing '{occ_class}' with multiple feature combinations...")
            df_occ = self.df_cleaned[self.df_cleaned['OCC_CLS'] == occ_class][features_extended].dropna().copy()

            if len(df_occ) > 10:
                occ_results = {
                    'total_buildings': len(df_occ),
                    'feature_combinations': {}
                }

                for combo in feature_combos:
                    print(f"    Computing clustering for {occ_class} with feature combo: {combo}")
                    combo_results = {}

                    for k in range(2, 8):
                        result = self._perform_clustering_with_features(df_occ, combo, k)
                        if result:
                            combo_results[k] = result

                    if combo_results:
                        occ_results['feature_combinations'][combo] = combo_results

                occupancy_clusters[occ_class] = occ_results

        return occupancy_clusters

    def process_occupancy_clusters(self):
        """Keep original method for backward compatibility"""
        print("Processing occupancy-specific clusters (original method)...")
        occupancy_clusters = {}
        features = ['SQMETERS', 'HEIGHT_USED', 'year_built', 'OCC_CLS', 'material_type', 'foundation_type',
                    'Est GFA sqmeters']


        # First, process for "all" classes
        print("  Processing 'all'...")
        df_all = self.df_cleaned[features].dropna().copy()
        k_results_all = self._get_cluster_stats_for_df(df_all)
        if k_results_all:
            occupancy_clusters['all'] = {
                'total_buildings': len(df_all),
                'k_values': k_results_all
            }

        # Then, process for each individual occupancy class
        for occ_class in self.df_cleaned['OCC_CLS'].unique():
            print(f"  Processing '{occ_class}'...")
            df_occ = self.df_cleaned[self.df_cleaned['OCC_CLS'] == occ_class][features].dropna().copy()

            k_results_occ = self._get_cluster_stats_for_df(df_occ)
            if k_results_occ:
                occupancy_clusters[occ_class] = {
                    'total_buildings': len(df_occ),
                    'k_values': k_results_occ
                }

        return occupancy_clusters

    def process_materials_foundation(self):
        """Process building materials and foundation data with occupancy breakdown AND Est GFA"""
        print("Processing materials and foundation data with occupancy breakdown and Est GFA...")

        # Process real data with occupancy breakdown and Est GFA
        materials_data = {}

        for filter_type, df_filtered in [
            ('all', self.df_cleaned),
            ('pre1940', self.df_cleaned[self.df_cleaned['year_built'] < 1940]),
            ('post1940', self.df_cleaned[self.df_cleaned['year_built'] >= 1940])
        ]:
            # Create contingency table for counts
            contingency = pd.crosstab(
                df_filtered['material_type'],
                df_filtered['foundation_type']
            )

            # Create contingency table for Est GFA
            area_contingency = pd.crosstab(
                df_filtered['material_type'],
                df_filtered['foundation_type'],
                values=df_filtered['Est GFA sqmeters'],
                aggfunc='sum'
            ).fillna(0)

            # Calculate occupancy breakdown for each material/foundation combination
            occupancy_breakdown = {}

            for mat in contingency.index:
                for found in contingency.columns:
                    # Get all buildings with this material/foundation combo
                    mask = (df_filtered['material_type'] == mat) & (df_filtered['foundation_type'] == found)
                    combo_buildings = df_filtered[mask]

                    if len(combo_buildings) > 0:
                        # Get occupancy counts and areas for this combination
                        occ_counts = combo_buildings['OCC_CLS'].value_counts()
                        occ_areas = combo_buildings.groupby('OCC_CLS')['Est GFA sqmeters'].sum()

                        key = f"{mat}_{found}"
                        occupancy_breakdown[key] = {
                            'total': len(combo_buildings),
                            'total_area': float(combo_buildings['Est GFA sqmeters'].sum()),
                            'occupancy_counts': occ_counts.to_dict(),
                            'occupancy_areas': occ_areas.to_dict()
                        }

            materials_data[filter_type] = {
                'matrix': contingency.values.tolist(),
                'area_matrix': area_contingency.values.tolist(),
                'materials': contingency.index.tolist(),
                'foundations': contingency.columns.tolist(),
                'occupancy_breakdown': occupancy_breakdown
            }

        return materials_data

    def process_soil_analysis(self):
        """
        Process all soil-related data.
        This includes mapping numerical engineering properties to categorical labels,
        calculating statistics for various soil features, and performing risk analysis.
        Enhanced: Now includes compname analysis
        """
        print("Processing soil data analysis...")

        # --- START: New code block for mapping numerical 'eng_property' to categories ---
        # Check if the 'eng_property' column exists and contains numeric data before attempting to map it.
        if 'eng_property' in self.df_cleaned.columns and pd.api.types.is_numeric_dtype(self.df_cleaned['eng_property']):
            print("  Mapping numerical engineering properties to categories based on defined ranges...")

            # Define the bin edges for the ranges. Using -inf and inf ensures all values are included.
            # You can adjust these bin edges based on your data's specific meaning.
            # Example ranges: (-inf, 0.17], (0.17, 0.24], (0.24, 0.32], (0.32, inf]
            bins = [-float('inf'), 0.17, 0.24, 0.32, float('inf')]

            # Define the string labels that correspond to each bin.
            labels = ['Favorable', 'Fair', 'Poor', 'Very poor']

            # Use the pandas 'cut' function to segment the data into the bins and assign the appropriate label.
            # This overwrites the original numeric 'eng_property' column with the new categorical data.
            self.df_cleaned['eng_property'] = pd.cut(self.df_cleaned['eng_property'], bins=bins, labels=labels,
                                                     right=True)
        # --- END: New code block ---

        soil_columns = ['drainagecl', 'wtdepannmin', 'flodfreqcl', 'eng_property',
                        'compname', 'comppct_r', 'MUSYM', 'mukey', 'LONGITUDE', 'LATITUDE']

        # Check which soil-related columns exist in the dataframe.
        existing_soil_cols = [col for col in soil_columns if col in self.df_cleaned.columns]


        # Initialize the dictionary to hold all so# Step 3: Track and remove missing OCC_CLSl analysis results.
        soil_analysis = {
            'drainage_class_stats': {},
            'flooding_freq_stats': {},
            'water_table_stats': {},
            'engineering_property_stats': {},
            'compname_stats': {},  # NEW: Added compname statistics
            'soil_by_occupancy': {},
            'spatial_distribution': [],
            'soil_risk_analysis': {}
        }

        if 'drainagecl' in self.df_cleaned.columns:
            drainage_counts = self.df_cleaned['drainagecl'].value_counts(dropna=False)
            counts_dict = drainage_counts.to_dict()
            if np.nan in counts_dict:
                nan_val = counts_dict.pop(np.nan)
                counts_dict['NaN (Missing)'] = nan_val

            soil_analysis['drainage_class_stats'] = {
                'counts': counts_dict,
                'percentages': {k: v / len(self.df_cleaned) * 100 for k, v in counts_dict.items()}
            }

        # Calculate flooding frequency statistics if the column exists.
        if 'flodfreqcl' in self.df_cleaned.columns:
            flood_counts = self.df_cleaned['flodfreqcl'].value_counts(dropna=False)
            counts_dict = flood_counts.to_dict()
            if np.nan in counts_dict:
                nan_val = counts_dict.pop(np.nan)
                counts_dict['NaN (Missing)'] = nan_val

            soil_analysis['flooding_freq_stats'] = {
                'counts': counts_dict,
                'percentages': {k: v / len(self.df_cleaned) * 100 for k, v in counts_dict.items()}
            }

        # Calculate water table depth statistics if the column exists.
        if 'wtdepannmin' in self.df_cleaned.columns:
            water_table = self.df_cleaned['wtdepannmin'].dropna()
            soil_analysis['water_table_stats'] = {
                'mean': float(water_table.mean()),
                'median': float(water_table.median()),
                'std': float(water_table.std()),
                'min': float(water_table.min()),
                'max': float(water_table.max()),
                'q25': float(water_table.quantile(0.25)),
                'q75': float(water_table.quantile(0.75))
            }

        # Calculate engineering property statistics if the column exists.
        if 'eng_property' in self.df_cleaned.columns:
            eng_counts = self.df_cleaned['eng_property'].value_counts(dropna=False)
            counts_dict = eng_counts.to_dict()
            if np.nan in counts_dict:
                nan_val = counts_dict.pop(np.nan)
                counts_dict['NaN (Missing)'] = nan_val

            soil_analysis['engineering_property_stats'] = {
                'counts': counts_dict,
                'percentages': {k: v / len(self.df_cleaned) * 100 for k, v in counts_dict.items()}
            }

        # NEW: Calculate compname statistics if the column exists
        if 'compname' in self.df_cleaned.columns:
            comp_counts = self.df_cleaned['compname'].value_counts(dropna=False)
            counts_dict = comp_counts.to_dict()
            nan_val = None
            if np.nan in counts_dict:
                nan_val = counts_dict.pop(np.nan)

            # Get top 20 most common soil component names from the non-NaN data
            top_comp_dict = dict(sorted(counts_dict.items(), key=lambda item: item[1], reverse=True)[:20])

            # Re-add the NaN count if it exists
            if nan_val is not None:
                top_comp_dict['NaN (Missing)'] = nan_val

            soil_analysis['compname_stats'] = {
                'counts': top_comp_dict,
                'percentages': {k: v / len(self.df_cleaned) * 100 for k, v in top_comp_dict.items()},
                'total_unique': len(comp_counts),
                'top_20_coverage': (sum(top_comp_dict.values()) - (nan_val or 0)) / len(self.df_cleaned) * 100
            }

        # Group soil properties by occupancy class.
        for occ_class in self.df_cleaned['OCC_CLS'].unique():
            occ_data = self.df_cleaned[self.df_cleaned['OCC_CLS'] == occ_class]
            occ_soil_stats = {}

            if 'drainagecl' in occ_data.columns:
                counts = occ_data['drainagecl'].value_counts(dropna=False)
                counts_dict = counts.to_dict()
                if np.nan in counts_dict:
                    counts_dict['NaN (Missing)'] = counts_dict.pop(np.nan)
                occ_soil_stats['drainage_distribution'] = counts_dict

            if 'flodfreqcl' in occ_data.columns:
                counts = occ_data['flodfreqcl'].value_counts(dropna=False)
                counts_dict = counts.to_dict()
                if np.nan in counts_dict:
                    counts_dict['NaN (Missing)'] = counts_dict.pop(np.nan)
                occ_soil_stats['flooding_distribution'] = counts_dict

            if 'eng_property' in occ_data.columns:
                counts = occ_data['eng_property'].value_counts(dropna=False)
                counts_dict = counts.to_dict()
                if np.nan in counts_dict:
                    counts_dict['NaN (Missing)'] = counts_dict.pop(np.nan)
                occ_soil_stats['engineering_distribution'] = counts_dict

            if 'compname' in occ_data.columns:
                counts = occ_data['compname'].value_counts(dropna=False)
                counts_dict = counts.to_dict()
                nan_val = None
                if np.nan in counts_dict:
                    nan_val = counts_dict.pop(np.nan)

                top_10_dict = dict(sorted(counts_dict.items(), key=lambda item: item[1], reverse=True)[:10])

                if nan_val is not None:
                    top_10_dict['NaN (Missing)'] = nan_val
                occ_soil_stats['compname_distribution'] = top_10_dict

            if 'wtdepannmin' in occ_data.columns:
                water_table_occ = occ_data['wtdepannmin'].dropna()
                if len(water_table_occ) > 0:
                    occ_soil_stats['water_table_stats'] = {
                        'mean': float(water_table_occ.mean()),
                        'median': float(water_table_occ.median()),
                        'std': float(water_table_occ.std())
                    }
            soil_analysis['soil_by_occupancy'][occ_class] = occ_soil_stats

        # Prepare a sample of data for the spatial map visualization.
        if 'LONGITUDE' in self.df_cleaned.columns and 'LATITUDE' in self.df_cleaned.columns:
            sample_size = min(75000, len(self.df_cleaned))
            spatial_sample = self.df_cleaned.sample(n=sample_size, random_state=337)

            for _, row in spatial_sample.iterrows():
                point_data = {
                    'lon': float(row['LONGITUDE']) if pd.notna(row['LONGITUDE']) else None,
                    'lat': float(row['LATITUDE']) if pd.notna(row['LATITUDE']) else None,
                    'occupancy': row['OCC_CLS'],
                    'year_built': int(row['year_built']),
                    'area': float(row['Est GFA sqmeters'])
                }
                if 'drainagecl' in row and pd.notna(row['drainagecl']):
                    point_data['drainage'] = row['drainagecl']
                if 'flodfreqcl' in row and pd.notna(row['flodfreqcl']):
                    point_data['flooding'] = row['flodfreqcl']
                if 'eng_property' in row and pd.notna(row['eng_property']):
                    point_data['eng_property'] = row['eng_property']
                if 'wtdepannmin' in row and pd.notna(row['wtdepannmin']):
                    point_data['water_table'] = float(row['wtdepannmin'])
                if 'compname' in row and pd.notna(row['compname']):
                    point_data['compname'] = row['compname']  # NEW: Added compname to spatial data

                if point_data['lon'] is not None and point_data['lat'] is not None:
                    soil_analysis['spatial_distribution'].append(point_data)

        # Perform risk analysis based on high-risk soil properties.
        if 'drainagecl' in self.df_cleaned.columns and 'flodfreqcl' in self.df_cleaned.columns:
            high_risk_drainage = ['Poorly drained', 'Very poorly drained']
            high_risk_flooding = ['High']

            high_risk_buildings = self.df_cleaned[
                (self.df_cleaned['drainagecl'].isin(high_risk_drainage)) |
                (self.df_cleaned['flodfreqcl'].isin(high_risk_flooding))
                ]

            soil_analysis['soil_risk_analysis'] = {
                'high_risk_count': len(high_risk_buildings),
                'high_risk_percentage': round(len(high_risk_buildings) / len(self.df_cleaned) * 100, 2),
                'high_risk_by_occupancy': high_risk_buildings['OCC_CLS'].value_counts().to_dict(),
                'high_risk_avg_year': int(high_risk_buildings['year_built'].mean()) if len(
                    high_risk_buildings) > 0 else 0,
                'high_risk_total_area': float(high_risk_buildings['Est GFA sqmeters'].sum())
            }

        return soil_analysis

    def calculate_nsi_data_sources(self):
        """Return hardcoded NSI methodology statistics"""
        print("Returning NSI data source methodology...")

        # These are fixed values representing NSI dataset methodology
        # Not specific to your MA dataset
        nsi_stats = {
            'methodology': 'NSI Dataset Construction',
            'note': 'These values represent the general NSI dataset methodology, not this specific MA subset'
        }

        return nsi_stats

    def get_cluster_analysis(self):
        """Get cluster analysis results"""
        print("Analyzing clusters...")

        # --- REVERT: This function should use 'Est GFA sqmeters' for the VISUALIZATION stats ---
        cluster_analysis = self.df_cluster.groupby('cluster').agg({
            'Est GFA sqmeters': ['mean', 'median', 'std'],
            'year_built': ['mean', 'median', 'std'],
            'OCC_CLS': [('count', 'size'), ('most_common', lambda x: x.value_counts().index[0])]
        })

        # Flatten column names
        cluster_analysis.columns = ['_'.join(col).strip() for col in cluster_analysis.columns]

        # Convert to list of dictionaries
        clusters = []
        for cluster_id in cluster_analysis.index:
            row = cluster_analysis.loc[cluster_id]
            clusters.append({
                'cluster_id': int(cluster_id),
                'count': int(row['OCC_CLS_count']),
                'most_common_occ': row['OCC_CLS_most_common'],
                'area_mean': float(row['Est GFA sqmeters_mean']),
                'area_median': float(row['Est GFA sqmeters_median']),
                'area_std': float(row['Est GFA sqmeters_std']) if not pd.isna(row['Est GFA sqmeters_std']) else 0,
                'year_mean': int(row['year_built_mean']),
                'year_median': int(row['year_built_median']),
                'year_std': float(row['year_built_std']) if not pd.isna(row['year_built_std']) else 0
            })

        return clusters

    def prepare_enhanced_samples(self):
        """
        Create samples with pre-computed clusters for all feature combinations
        Returns DataFrames for export
        """
        print("Creating enhanced samples with multi-dimensional clustering...")

        # --- FINAL FIX: Include 'Est GFA sqmeters' for JS visualizations that use samples ---
        features = ['Est GFA sqmeters', 'SQMETERS', 'HEIGHT_USED','PRED_HEIGHT', 'year_built',
                    'OCC_CLS', 'material_type', 'foundation_type', 'Assumed height']

        # Add soil features if they exist
        soil_features = ['drainagecl', 'flodfreqcl', 'eng_property', 'wtdepannmin', 'compname', 'LONGITUDE', 'LATITUDE']
        for sf in soil_features:
            if sf in self.df_cleaned.columns:
                features.append(sf)

        df_for_samples = self.df_cleaned[features].dropna(
            subset=['SQMETERS', 'HEIGHT_USED', 'year_built', 'OCC_CLS']
        ).copy()

        # Outlier removal should use GFA to match original intent
        area_threshold = df_for_samples['Est GFA sqmeters'].quantile(0.99999)
        df_for_samples = df_for_samples[df_for_samples['Est GFA sqmeters'] < area_threshold]

        # Create random sample
        random_sample_size = min(75000, len(df_for_samples))
        random_sample_df = df_for_samples.sample(n=random_sample_size, random_state=337).copy()

        # Create balanced sample
        SAMPLES_PER_CLASS = 2500
        balanced_sample_df = df_for_samples.groupby('OCC_CLS', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), SAMPLES_PER_CLASS), random_state=337)
        ).copy()

        # (The rest of the function in your file remains the same)
        random_sample_df = random_sample_df.reset_index(drop=True)
        balanced_sample_df = balanced_sample_df.reset_index(drop=True)

        for sample_df, sample_name in [(random_sample_df, 'random'), (balanced_sample_df, 'balanced')]:
            print(f"  Performing REAL clustering on {sample_name} sample...")
            feature_combos = ['base', 'material', 'foundation', 'both']
            for combo in feature_combos:
                print(f"    - Clustering with feature combo: {combo}")
                for k in range(2, 10):
                    cluster_assignments = self._get_cluster_assignments_for_df(sample_df, combo, k)
                    sample_df[f'cluster_{combo}_k{k}'] = cluster_assignments
            print(f"  - Finalizing cluster columns for {sample_name} sample...")
            for k in range(2, 10):
                if f'cluster_base_k{k}' in sample_df.columns:
                    sample_df[f'cluster_k{k}'] = sample_df[f'cluster_base_k{k}']
            if 'cluster_base_k7' in sample_df.columns:
                sample_df['cluster'] = sample_df['cluster_base_k7']
            else:
                sample_df['cluster'] = None

        print(f"  Random sample size: {len(random_sample_df)}")
        print(f"  Balanced sample size: {len(balanced_sample_df)}")
        return random_sample_df, balanced_sample_df

    def clean_for_json(self, obj):
        """Recursively clean data for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self.clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.clean_for_json(item) for item in obj]
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        elif isinstance(obj, (np.floating, np.complexfloating)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return self.clean_for_json(obj.tolist())
        elif pd.isna(obj):
            return None
        else:
            return obj

    def process_clf_data(self, csv_path='USASTR_MA.csv', output_path='clf_data.json'):
        """
        Processes the CLF dataset (USASTR_MA.csv) for the new dashboard section.
        NOW INCLUDES:
        - 4 heatmap variations:
          1. Mapped Material vs. Foundation (by Count)
          2. Mapped Material vs. Foundation (by Est GFA sqmeters)
          3. Structural System vs. Foundation (by Count)
          4. Structural System vs. Foundation (by Est GFA sqmeters)
        """
        print(f"\nProcessing CLF data from {csv_path}...")

        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"Error: CLF file not found at {csv_path}. Skipping CLF processing.")
            return
        except Exception as e:
            print(f"Error reading {csv_path}: {e}. Skipping CLF processing.")
            return

        # 1. Prepare data for the 2D scatter plot
        # (No change here from previous step)
        scatter_cols = ['Est GFA sqmeters', 'mass_total', 'gwp_a_to_c', 'OCC_CLS', 'material_type', 'str_sys_summary']
        df_scatter = df[scatter_cols].dropna()
        scatter_data = {col: df_scatter[col].tolist() for col in df_scatter.columns}
        print(f"  Processed {len(df_scatter)} records for CLF scatter plot.")

        # 2. Prepare data for all 4 heatmap variations
        heatmap_cols = ['material_type', 'general_fnd_type', 'str_sys_summary', 'Est GFA sqmeters']

        # We need to ensure the categorical columns are present, but GFA can be NaN if we're just counting
        df_heatmap = df[heatmap_cols].dropna(subset=['material_type', 'general_fnd_type', 'str_sys_summary'])

        print(f"  Processing {len(df_heatmap)} records for CLF heatmaps.")

        def _create_heatmap_dict(df_crosstab, is_gfa=False):
            """Helper to convert crosstab df to dict format"""
            # Re-align GFA dataframe if it's sparse (missing combinations)
            if is_gfa:
                # Get the full index/columns from a count crosstab to ensure alignment
                if 'material_type' in df_crosstab.index.name:
                    base_df = pd.crosstab(df_heatmap['material_type'], df_heatmap['general_fnd_type'])
                else:
                    base_df = pd.crosstab(df_heatmap['str_sys_summary'], df_heatmap['general_fnd_type'])

                # Reindex to match the full matrix, filling missing combos with 0
                df_crosstab = df_crosstab.reindex(index=base_df.index, columns=base_df.columns).fillna(0)

            return {
                'z': df_crosstab.values.tolist(),
                'x': df_crosstab.columns.tolist(),  # X-axis labels (general_fnd_type)
                'y': df_crosstab.index.tolist()  # Y-axis labels (material_type or str_sys_summary)
            }

        # --- Create all 4 heatmap dataframes ---

        # 1. Mapped Material vs. Foundation (Count)
        df_mat_fnd_count = pd.crosstab(df_heatmap['material_type'], df_heatmap['general_fnd_type'])

        # 2. Mapped Material vs. Foundation (GFA)
        df_mat_fnd_gfa = pd.crosstab(df_heatmap['material_type'], df_heatmap['general_fnd_type'],
                                     values=df_heatmap['Est GFA sqmeters'], aggfunc='sum')

        # 3. Structural System vs. Foundation (Count)
        df_str_fnd_count = pd.crosstab(df_heatmap['str_sys_summary'], df_heatmap['general_fnd_type'])

        # 4. Structural System vs. Foundation (GFA)
        df_str_fnd_gfa = pd.crosstab(df_heatmap['str_sys_summary'], df_heatmap['general_fnd_type'],
                                     values=df_heatmap['Est GFA sqmeters'], aggfunc='sum')

        # --- Convert to dicts for JSON export ---
        heatmap_material_count = _create_heatmap_dict(df_mat_fnd_count)
        heatmap_material_gfa = _create_heatmap_dict(df_mat_fnd_gfa, is_gfa=True)
        heatmap_struct_count = _create_heatmap_dict(df_str_fnd_count)
        heatmap_struct_gfa = _create_heatmap_dict(df_str_fnd_gfa, is_gfa=True)

        print(f"  Processed 4 heatmap variations (Material/Struct vs. GFA/Count).")

        # 3. Combine and export to JSON
        output_data = {
            'scatter_data': scatter_data,
            'heatmap_material_count': heatmap_material_count,
            'heatmap_material_gfa': heatmap_material_gfa,
            'heatmap_struct_count': heatmap_struct_count,
            'heatmap_struct_gfa': heatmap_struct_gfa,
        }

        # Use the existing clean_for_json method (if defined in the class)
        if hasattr(self, 'clean_for_json'):
            output_data = self.clean_for_json(output_data)

        try:
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)  # indent=2 for readability
            print(f"Successfully saved CLF data to {output_path}")
        except Exception as e:
            print(f"Error writing CLF JSON to {output_path}: {e}")

    def export_to_json(self, output_path='building_data.json'):
        """Export all processed data to JSON - Split into main and multiple sample files"""
        print("Exporting data to JSON (split into multiple files)...")

        # Get elbow scores
        k_range, wcss = self.calculate_elbow_scores()

        # Pre-calculate enhanced occupancy clusters
        occupancy_clusters_enhanced = self.process_occupancy_clusters_enhanced()

        # Also keep original occupancy clusters for backward compatibility
        occupancy_clusters_data = self.process_occupancy_clusters()

        # Get overview occupancy counts
        overview_occupancy_counts = self.get_overview_occupancy_counts()

        mix_sc_distribution = self.process_mix_sc_distribution()

        # Process soil analysis with compname
        soil_analysis_data = self.process_soil_analysis()

        # OCC type sankey
        occupancy_hierarchy_sankey = self.process_occupancy_hierarchy()

        # Calculate NSI data source statistics
        nsi_data_sources = self.calculate_nsi_data_sources()


        hierarchical_distribution = self.process_hierarchical_distribution()

        # NEW: full-pop aggregation for the new Year→Occ→Mat→Found→Soil Sankey
        year_occ_flow = self.process_year_occ_mat_found_soil_flow()
        occ_cls_occ_dict_sankey = self.process_occ_cls_to_occdict_sankey()
        # Get enhanced samples as DataFrames
        random_sample_df, balanced_sample_df = self.prepare_enhanced_samples()

        # Prepare MAIN export data (without samples)
        main_data = {
            'metadata': {
                'total_buildings': len(self.df_cleaned),
                'date_processed': datetime.now().isoformat(),
                'source_file': self.csv_path,
                'version': '3.2',  # Version 3.2 includes compname and data flow analysis
                'has_samples_file': True,
                'samples_split': True,
                'samples_files': []
            },
            'hierarchical_distribution': hierarchical_distribution,
            'occ_cls_occ_dict_sankey': occ_cls_occ_dict_sankey,
            'year_occ_flow': year_occ_flow,
            'summary_stats': {
                'total_buildings': len(self.df_cleaned),
                'avg_year_built': int(self.df_cleaned['year_built'].mean()),
                'avg_area_sqm': float(self.df_cleaned['Est GFA sqmeters'].dropna().mean()),
                'min_year': int(self.df_cleaned['year_built'].min()),
                'max_year': int(self.df_cleaned['year_built'].max()),
                'occupancy_classes': sorted(self.df_cleaned['OCC_CLS'].unique().tolist())
            },
            'overview_occupancy_counts': overview_occupancy_counts,
            'mix_sc_distribution': mix_sc_distribution,
            'clustering': {
                'elbow_k_values': k_range,
                'elbow_wcss_values': wcss,
                'clusters': self.get_cluster_analysis()
            },
            'temporal_data': self.process_temporal_data(),
            'pre1940': self.process_pre1940_data(),
            'post1940': self.process_post1940_data(),
            'occupancy_clusters': occupancy_clusters_data,
            'occupancy_clusters_enhanced': occupancy_clusters_enhanced,
            'materials_foundation': self.process_materials_foundation(),
            'soil_analysis': soil_analysis_data,  # Now includes compname analysis
            'occupancy_hierarchy_sankey': occupancy_hierarchy_sankey,
            'data_flow_stats': self.data_flow_stats,  # NEW: Data flow statistics
            'nsi_data_sources': nsi_data_sources  # NEW: NSI data source statistics
        }

        # Clean main data for JSON
        main_data = self.clean_for_json(main_data)

        # Split samples into chunks
        CHUNK_SIZE = 5000

        # Convert to list for chunking and clean for JSON
        random_samples_list = [self.clean_for_json(row) for row in random_sample_df.to_dict(orient='records')]
        balanced_samples_list = [self.clean_for_json(row) for row in balanced_sample_df.to_dict(orient='records')]

        # Split random samples into chunks
        random_chunks = [random_samples_list[i:i + CHUNK_SIZE]
                         for i in range(0, len(random_samples_list), CHUNK_SIZE)]

        # Split balanced samples into chunks
        balanced_chunks = [balanced_samples_list[i:i + CHUNK_SIZE]
                           for i in range(0, len(balanced_samples_list), CHUNK_SIZE)]

        sample_files_info = []
        total_samples_size = 0

        # Save random sample chunks
        for i, chunk in enumerate(random_chunks):
            filename = output_path.replace('.json', f'_samples_random_{i + 1}.json')
            chunk_data = {
                'metadata': {
                    'type': 'random',
                    'chunk_index': i + 1,
                    'total_chunks': len(random_chunks),
                    'chunk_size': len(chunk),
                    'date_generated': datetime.now().isoformat()
                },
                'samples': chunk
            }

            with open(filename, 'w') as f:
                json.dump(chunk_data, f, separators=(',', ':'))  # Compact format

            chunk_size_mb = len(json.dumps(chunk_data, separators=(',', ':'))) / 1024 / 1024
            total_samples_size += chunk_size_mb

            sample_files_info.append({
                'filename': filename.split('/')[-1],
                'type': 'random',
                'chunk_index': i + 1,
                'sample_count': len(chunk),
                'size_mb': round(chunk_size_mb, 2)
            })

            print(f"  Saved {filename} ({chunk_size_mb:.2f} MB, {len(chunk)} samples)")

        # Save balanced sample chunks
        for i, chunk in enumerate(balanced_chunks):
            filename = output_path.replace('.json', f'_samples_balanced_{i + 1}.json')
            chunk_data = {
                'metadata': {
                    'type': 'balanced',
                    'chunk_index': i + 1,
                    'total_chunks': len(balanced_chunks),
                    'chunk_size': len(chunk),
                    'date_generated': datetime.now().isoformat()
                },
                'samples': chunk
            }

            with open(filename, 'w') as f:
                json.dump(chunk_data, f, separators=(',', ':'))

            chunk_size_mb = len(json.dumps(chunk_data, separators=(',', ':'))) / 1024 / 1024
            total_samples_size += chunk_size_mb

            sample_files_info.append({
                'filename': filename.split('/')[-1],
                'type': 'balanced',
                'chunk_index': i + 1,
                'sample_count': len(chunk),
                'size_mb': round(chunk_size_mb, 2)
            })

            print(f"  Saved {filename} ({chunk_size_mb:.2f} MB, {len(chunk)} samples)")

        # Update main data with sample files info
        main_data['metadata']['samples_files'] = sample_files_info
        main_data['metadata']['total_random_samples'] = len(random_samples_list)
        main_data['metadata']['total_balanced_samples'] = len(balanced_samples_list)
        main_data['metadata']['random_chunks'] = len(random_chunks)
        main_data['metadata']['balanced_chunks'] = len(balanced_chunks)

        # Save main data
        with open(output_path, 'w') as f:
            json.dump(main_data, f, indent=2)

        main_size = len(json.dumps(main_data)) / 1024 / 1024

        print(f"\n{'=' * 60}")
        print(f"Export Complete!")
        print(f"{'=' * 60}")
        print(f"Main data exported to: {output_path} ({main_size:.2f} MB)")
        print(f"Sample files created: {len(sample_files_info)} files")
        print(f"  - Random samples: {len(random_chunks)} files ({len(random_samples_list)} total samples)")
        print(f"  - Balanced samples: {len(balanced_chunks)} files ({len(balanced_samples_list)} total samples)")
        print(f"Total samples size: {total_samples_size:.2f} MB")
        print(f"Average file size: {total_samples_size / len(sample_files_info):.2f} MB")
        print(f"Soil analysis data included: Yes (with compname analysis)")
        print(f"Data flow statistics included: Yes")
        print(f"NSI data source analysis included: Yes")

        # Check if any file exceeds 25MB
        for file_info in sample_files_info:
            if file_info['size_mb'] > 25:
                print(f"WARNING: {file_info['filename']} exceeds 25MB ({file_info['size_mb']} MB)")
                print(f"Consider reducing CHUNK_SIZE to {int(CHUNK_SIZE * 20 / file_info['size_mb'])}")

        return main_data

def main():
    """Main processing function"""
    print("="*60)
    print("Massachusetts Building Data Processing - Multi-dimensional Enhanced Version with Soil and Data Flow Analysis")
    print("="*60)

    # Initialize processor
    processor = BuildingDataProcessor('ma_structures_with_demolition_FINAL.csv')

    # Process data
    processor.load_data()
    processor.clean_data()
    processor.resolve_unclassified_from_occdict()
    processor.recalculate_mix_sc_for_reclassified()
    processor.prepare_clustering_data(remove_outliers=False)
    processor.perform_clustering(n_clusters=7)

    # Export to JSON
    export_data = processor.export_to_json('building_data.json')

    print("\n" + "=" * 60)
    print("Processing CLF (USASTR_MA.csv) Data...")

    processor.process_clf_data('USASTR_MA.csv', 'clf_data.json')



    raw_height_numeric = pd.to_numeric(processor.df['HEIGHT'], errors='coerce')
    raw_gfa_negative_count = (processor.df['Est GFA sqmeters'] < 0).sum()
    raw_height_negative_count = (raw_height_numeric < 0).sum()


    cleaned_gfa_negative_count = (processor.df_cleaned['Est GFA sqmeters'] < 0).sum()


    cleaned_height_numeric = pd.to_numeric(processor.df_cleaned['HEIGHT'], errors='coerce')
    cleaned_height_negative_count = (cleaned_height_numeric < 0).sum()

    print("Data Quality Checks:")
    print(f"  Raw Data (Initial Load):")
    print(f"    - Records with Est GFA sqmeters < 0: {raw_gfa_negative_count:,}")
    print(f"    - Records with HEIGHT < 0:      {raw_height_negative_count:,}")
    print(f"  Cleaned Data (Final Dataset):")
    print(f"    - Records with Est GFA sqmeters < 0: {cleaned_gfa_negative_count:,} (expect 0)")
    print(f"    - Records with HEIGHT < 0:      {cleaned_height_negative_count:,}")
    print("-" * 60)



    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"Total buildings processed: {export_data['metadata']['total_buildings']:,}")
    print(f"Overview occupancy classes: {len(export_data.get('overview_occupancy_counts', {}))} types")
    print(f"Temporal data points: {len(export_data.get('temporal_data', []))}")
    print(f"Occupancy-specific clusters: {len(export_data.get('occupancy_clusters', {}))} classes")
    print(f"Enhanced clusters with features: {len(export_data.get('occupancy_clusters_enhanced', {}))} classes")
    print(f"Soil analysis included: Yes (with compname analysis)")
    print(f"Data flow analysis included: Yes")
    print("\nData exported to: building_data.json and building_data_samples_*.json files")
    print("You can now open the updated HTML dashboard to visualize the data including all new analyses")

if __name__ == "__main__":
    main()