import collections, awkward as ak, numpy as np
import os
import uproot
from coffea import processor
from coffea.analysis_tools import Weights

# functions to load SFs, corrections
from BTVNanoCommissioning.utils.correction import (
    load_lumi,
    load_SF,
    common_shifts,
    weight_manager,
)

# user helper function
from BTVNanoCommissioning.helpers.func import (
    flatten,
    update,
    uproot_writeable,
    dump_lumi,
)
from BTVNanoCommissioning.helpers.update_branch import missing_branch

## load histograms & selctions for this workflow
from BTVNanoCommissioning.utils.histogrammer import histogrammer, histo_writter
from BTVNanoCommissioning.utils.array_writer import array_writer
from BTVNanoCommissioning.utils.selection import (
    HLT_helper,
    jet_id,
    mu_idiso,
    ele_cuttightid,
)


class NanoProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year="2022",
        campaign="Summer22Run3",
        name="",
        isSyst=False,
        isArray=False,
        noHist=False,
        chunksize=75000,
    ):
        self._year = year
        self._campaign = campaign
        self.name = name
        self.isSyst = isSyst
        self.isArray = isArray
        self.noHist = noHist
        self.lumiMask = load_lumi(self._campaign)
        self.chunksize = chunksize
        ## Load corrections
        self.SF_map = load_SF(self._year, self._campaign)

    @property
    def accumulator(self):
        return self._accumulator

    ## Apply corrections on momentum/mass on MET, Jet, Muon
    def process(self, events):
        events = missing_branch(events)
        shifts = common_shifts(self, events)

        return processor.accumulate(
            self.process_shift(update(events, collections), name)
            for collections, name in shifts
        )

    ## Processed events per-chunk, made selections, filled histogram, stored root files
    def process_shift(self, events, shift_name):
        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")
        ######################
        #  Create histogram  # : Get the histogram dict from `histogrammer`
        ######################
        _hist_event_dict = (
            {"": None}
            if self.noHist
            else histogrammer(events, "qgtag_DY")  # this is the place to modify
        )

        output = {
            "sumw": processor.defaultdict_accumulator(float),
            **_hist_event_dict,
        }
        if shift_name is None:
            if isRealData:
                output["sumw"] = len(events)
            else:
                output["sumw"] = ak.sum(events.genWeight)

        ####################
        #    Selections    #
        ####################
        ## Lumimask
        req_lumi = np.ones(len(events), dtype="bool")
        if isRealData:
            req_lumi = self.lumiMask(events.run, events.luminosityBlock)
        # only dump for nominal case
        if shift_name is None:
            output = dump_lumi(events[req_lumi], output)
        ##====> start here, make your customize modification
        ## HLT
        if self._year == "2016":
            triggers = [
                "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ"
            ]
        elif self._year in ["2017", "2018"]:
            triggers = [
                "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8",
            ]
        else:
            triggers = [
                "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8"
            ]
        req_trig = HLT_helper(events, triggers)

        # events = events[req_trig & req_lumi]
        event_level = req_trig & req_lumi

        ##### Add some selections
        ## Muon cuts
        # muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2

        muon_sel = (events.Muon.pt > 15) & (mu_idiso(events, self._campaign)) # muon pt > 20 GeV in Run2?
        event_mu = ak.mask(events.Muon, muon_sel)
        event_mu = ak.pad_none(event_mu, 2)
        events.Muon = ak.pad_none(events.Muon, 2) # Make sure that the shape is consistent

        req_muon = ak.count(event_mu.pt, axis=1) == 2
        event_level = event_level & req_muon

        req_muChrg = ak.where(req_muon, event_mu[:, 0].charge != event_mu[:, 1].charge, ak.zeros_like(req_muon))

        event_level = event_level & req_muChrg

        # Electron cut
        ele_sel = (events.Electron.pt > 15) & (ele_cuttightid(events, self._campaign))
        event_e = events.Electron[ele_sel]
        req_ele = ak.num(event_e.pt) == 0

        event_level = event_level & req_ele

        ## Jet cuts
        jet_sel = (events.Jet.pt > 30) & (jet_id(events, self._campaign))

        if self._year == "2016":
            jet_puid = events.Jet.puId >= 1
        elif self._year in ["2017", "2018"]:
            jet_puid = events.Jet.puId >= 4
        else:
            jet_puid = True

        # Jet isolation from muons
        # & deltaPhi to Z
        req_jetmu = ak.all(events.Jet.metric_table(event_mu) > 0.4, axis=2, mask_identity=True)
        jet_dphi = (event_mu[:, 0] + event_mu[:, 1]).delta_phi(events.Jet) > 2.7

        jet_sel = jet_sel & jet_puid & req_jetmu & jet_dphi
        event_jet = ak.mask(events.Jet, jet_sel)
        event_jet = ak.pad_none(event_jet, 2)
        events.Jet = ak.pad_none(events.Jet, 2) # Make sure that the shape is consistent
        
        req_jet = ak.count(event_jet.pt, axis=1) > 0

        event_level = event_level & req_jet

        ## MC only: require gen vertex to be close to reco vertex
        if "GenVtx_z" in events.fields:
            req_vtx = np.abs(events.GenVtx_z - events.PV_z) < 0.2
        else:
            req_vtx = ak.ones_like(events.run, dtype=bool)

        event_level = event_level & req_vtx
        
        ## Z candidate
        req_zmass = ak.where(req_muon, (event_mu[:, 0] + event_mu[:, 1]).mass > 71.2, ak.zeros_like(req_muon))
        req_zpt = ak.where(req_muon, (event_mu[:, 0] + event_mu[:, 1]).pt > 12, ak.zeros_like(req_muon))

        req_subjet = ak.where(ak.count(event_jet.pt, axis=1) > 1 & req_muon, event_jet[:, 1].pt / (event_mu[:, 0] + event_mu[:, 1]).pt < 1.0, ak.ones_like(req_muon))

        req_z = req_zmass & req_zpt 
        event_level = event_level & req_z & req_subjet

        ##<==== finish selection
        event_level = ak.fill_none(event_level, False)
        # Skip empty events -
        if len(events[event_level]) == 0:
            if self.isArray:
                array_writer(
                    self,
                    events[event_level],
                    events,
                    "nominal",
                    dataset,
                    isRealData,
                    empty=True,
                )
            return {dataset: output}

        ##===>  Ntuplization  : store custom information
        ####################
        # Selected objects # : Pruned objects with reduced event_level
        ####################
        # Keep the structure of events and pruned the object size
        pruned_ev = events[event_level]

        pruned_ev["SelJet"] = pruned_ev.Jet[:, 0]
        pruned_ev["PosMuon"] = pruned_ev.Muon[pruned_ev.Muon.charge > 0][:, 0]
        pruned_ev["NegMuon"] = pruned_ev.Muon[pruned_ev.Muon.charge < 0][:, 0]
        pruned_ev["ZCand"] = pruned_ev.Muon[:, 0] + pruned_ev.Muon[:, 1]
        pruned_ev["ZCand", "pt"] = pruned_ev["ZCand"].pt
        pruned_ev["ZCand", "eta"] = pruned_ev["ZCand"].eta
        pruned_ev["ZCand", "phi"] = pruned_ev["ZCand"].phi
        pruned_ev["ZCand", "mass"] = pruned_ev["ZCand"].mass
        pruned_ev["njet"] = ak.count(pruned_ev.Jet.pt, axis=1)

        ## <========= end: store custom objects

        ####################
        #     Output       #
        ####################
        # Configure SFs
        weights = weight_manager(pruned_ev, self.SF_map, self.isSyst)
        # Configure systematics
        if shift_name is None:
            systematics = ["nominal"] + list(weights.variations)
        else:
            systematics = [shift_name]
        if not isRealData:
            pruned_ev["weight"] = weights.weight()
            for ind_wei in weights.weightStatistics.keys():
                pruned_ev[f"{ind_wei}_weight"] = weights.partial_weight(
                    include=[ind_wei]
                )

        # Configure histograms
        if not self.noHist:
            output = histo_writter(
                pruned_ev, output, weights, systematics, self.isSyst, self.SF_map
            )
        # Output arrays
        if self.isArray:
            array_writer(self, pruned_ev, events, systematics[0], dataset, isRealData)

        return {dataset: output}

    ## post process, return the accumulator, compressed
    def postprocess(self, accumulator):
        return accumulator
