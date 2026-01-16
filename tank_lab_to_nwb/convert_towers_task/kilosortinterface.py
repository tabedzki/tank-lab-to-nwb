"""Custom KiloSort interface that adds electrode group information for multi-probe support."""

from neuroconv.datainterfaces import KiloSortSortingInterface


class KiloSortWithProbeInterface(KiloSortSortingInterface):
    """
    Custom KiloSort interface that properly handles electrode group assignment.
    
    This subclass extends KiloSortSortingInterface to add electrode_group information
    to units, enabling identification of which probe each unit came from in multi-probe
    recordings.
    
    Parameters
    ----------
    folder_path : str
        Path to the Kilosort output folder
    electrode_group_name : str, optional
        Name of the electrode group (probe) these units belong to
    **kwargs : dict
        Additional keyword arguments passed to KiloSortSortingInterface
    """
    
    # Override the ExtractorName to use the parent's extractor, not a custom one
    ExtractorName = "KiloSortSortingExtractor"
    
    def __init__(self, folder_path: str, electrode_group_name: str = None, **kwargs):
        # Store electrode_group_name before passing to parent
        self.electrode_group_name = electrode_group_name
        
        # Remove electrode_group_name from kwargs if it exists (shouldn't be passed to parent)
        kwargs.pop('electrode_group_name', None)
        
        # Call parent class __init__
        super().__init__(folder_path=folder_path, **kwargs)
    
    def add_to_nwbfile(self, nwbfile, metadata, **kwargs):
        """
        Add units to NWB file with electrode_group information.
        
        This method sets the electrode_group property on the sorting extractor units
        before calling the parent add_to_nwbfile method.
        """
        # Set electrode_group property on all units in the sorting extractor
        # This way, the parent interface will automatically include it in the units table
        if self.electrode_group_name is not None and self.electrode_group_name in nwbfile.electrode_groups:
            electrode_group_obj = nwbfile.electrode_groups[self.electrode_group_name]
            # Set electrode_group for each unit (using the electrode group object reference)
            try:
                for unit_id in self.sorting_extractor.get_unit_ids():
                    self.sorting_extractor.set_unit_property(
                        unit_id=unit_id,
                        property_name="electrode_group",
                        value=electrode_group_obj  # Use the actual ElectrodeGroup object
                    )
            except Exception as e:
                print(f"Warning: Could not set electrode_group property: {e}")
        
        # Call parent to add units (which will now include electrode_group property)
        super().add_to_nwbfile(nwbfile=nwbfile, metadata=metadata, **kwargs)
