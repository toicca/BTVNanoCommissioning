import collections, awkward as ak, numpy as np
import os
import uproot
from coffea import processor
from coffea.analysis_tools import Weights

# functions to load SFs, corrections
from BTVNanoCommissioning.utils.correction import (
    load_lumi,
    load_SF,
    eleSFs,
    muSFs,
    puwei,
    btagSFs,
    JME_shifts,
    Roccor_shifts,
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
from BTVNanoCommissioning.utils.histogrammer import histogrammer
from BTVNanoCommissioning.utils.array_writer import array_writer
from BTVNanoCommissioning.utils.selection import jet_id, mu_idiso, ele_cuttightid


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
        self.SF_map = load_SF(self._campaign)

    @property
    def accumulator(self):
        return self._accumulator

    ## Apply corrections on momentum/mass on MET, Jet, Muon
    def process(self, events):
        isRealData = not hasattr(events, "genWeight")

        events = missing_branch(events)
        shifts = []
        if "JME" in self.SF_map.keys():
            syst_JERC = self.isSyst
            if self.isSyst == "JERC_split":
                syst_JERC = "split"  # JEC splitted into 11 sources instead of JES_total
            shifts = JME_shifts(
                shifts, self.SF_map, events, self._campaign, isRealData, syst_JERC
            )
        else:
            if int(self._year) < 2020:
                shifts = [
                    ({"Jet": events.Jet, "MET": events.MET, "Muon": events.Muon}, None)
                ]
            else:
                shifts = [
                    (
                        {
                            "Jet": events.Jet,
                            "MET": events.PuppiMET,
                            "Muon": events.Muon,
                        },
                        None,
                    )
                ]
        if "roccor" in self.SF_map.keys():
            shifts = Roccor_shifts(shifts, self.SF_map, events, isRealData, False)
        else:
            shifts[0][0]["Muon"] = events.Muon

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
            {"": None} if self.noHist else histogrammer(events, "example")
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

        ## HLT
        if self._year == "2016":
            triggers = [
                "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ",
            ]
        elif self._year in ["2017", "2018"]:
            triggers = [
                "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8",
            ]
        else:
            triggers = [
                "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
            ]

        checkHLT = ak.Array([hasattr(events.HLT, _trig) for _trig in triggers])
        if ak.all(checkHLT == False):
            raise ValueError("HLT paths:", triggers, " are all invalid in", dataset)
        elif ak.any(checkHLT == False):
            print(np.array(triggers)[~checkHLT], " not exist in", dataset)
        trig_arrs = [
            events.HLT[_trig] for _trig in triggers if hasattr(events.HLT, _trig)
        ]
        req_trig = np.zeros(len(events), dtype="bool")
        for t in trig_arrs:
            req_trig = req_trig | t

        ##### Add some selections
        ## Muon cuts
        # muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2

        muon_sel = (events.Muon.pt > 15) & (mu_idiso(events, self._campaign)) & (np.abs(events.Muon.eta) < 2.4) & (events.Muon.pfRelIso04_all < 0.15) & (events.Muon.tightId)
        event_mu = events.Muon[muon_sel]
        req_muon = ak.num(event_mu.pt) == 2
        opp_charge = ak.sum(event_mu.charge, axis=1) == 0

        # Electron cut
        ele_sel = ((events.Electron.pt > 10) & (ele_cuttightid(events, self._campaign)) & (np.abs(events.Electron.eta) < 2.4)
                    & (
                        ((np.abs(events.Electron.eta) < 1.4442) & (np.abs(events.Electron.dz) < 0.1) & (np.abs(events.Electron.dxy) < 0.05))
                        | ((np.abs(events.Electron.eta) > 1.566) & (np.abs(events.Electron.dz) < 0.2) & (np.abs(events.Electron.dxy) < 0.1))
                        )
                    & (events.Electron.cutBased > 0)
                )
        event_e = events.Electron[ele_sel]
        req_ele = ak.num(event_e.pt) == 0

        ## Jet cuts
        jet_sel = (events.Jet.pt > 20) & (jet_id(events, self._campaign)) & (np.abs(events.Jet.eta) < 3.0)
        event_jet = events.Jet[jet_sel]
        req_jet = ak.num(event_jet) >= 1
        # Does this work to select jets that are not overlapping with muons?
        # >>>
        muon_iso = ak.all(events.Jet.metric_table(event_mu) > 0.4, axis=2, mask_identity=True)
        # <<<

        ## Other cuts

        ## Apply all selections
        event_level = (
            req_trig & req_lumi & req_jet & req_muon & req_ele & opp_charge & muon_iso
        )

        event_level = ak.fill_none(event_level, False)
        # Skip empty events
        if len(events[event_level]) == 0:
            return {dataset: output}

        ####################
        # Selected objects # : Pruned objects with reduced event_level
        ####################
        smu = event_mu[event_level]
        sjets = event_jet[event_level]
        sele = event_e[event_level]
        ####################
        # Weight & Geninfo # : Add weight to selected events
        ####################
        # create Weights object to save individual weights
        weights = Weights(len(events[event_level]), storeIndividual=True)
        if not isRealData:
            weights.add("genweight", events[event_level].genWeight)
            # par_flav = (sjets.partonFlavour == 0) & (sjets.hadronFlavour == 0)
            # genflavor = ak.values_astype(sjets.hadronFlavour + 1 * par_flav, int)
            # Load SFs
            if len(self.SF_map.keys()) > 0:
                syst_wei = (
                    True if self.isSyst != None else False
                )  # load systematic flag
                if "PU" in self.SF_map.keys():
                    puwei(
                        events[event_level].Pileup.nTrueInt,
                        self.SF_map,
                        weights,
                        syst_wei,
                    )
                if "MUO" in self.SF_map.keys():
                    muSFs(
                        smu, self.SF_map, weights, syst_wei, False
                    )  # input selected muon
                if "EGM" in self.SF_map.keys():
                    eleSFs(sele, self.SF_map, weights, syst_wei, False)
                if "BTV" in self.SF_map.keys():
                    # For BTV weight, you need to specify type
                    btagSFs(sjets, self.SF_map, weights, "DeepJetC", syst_wei)
                    btagSFs(sjets, self.SF_map, weights, "DeepJetB", syst_wei)
                    btagSFs(sjets, self.SF_map, weights, "DeepCSVB", syst_wei)
                    btagSFs(sjets, self.SF_map, weights, "DeepCSVC", syst_wei)
        else:
            genflavor = ak.zeros_like(sjets.pt, dtype=int)

        # Systematics information (add name of systematics)
        if shift_name is None:  # weight variations
            systematics = ["nominal"] + list(weights.variations)
        else:  # resolution/ scale variation would use the shift_name
            systematics = [shift_name]
        exclude_btv = [
            "DeepCSVC",
            "DeepCSVB",
            "DeepJetB",
            "DeepJetC",
        ]  # exclude b-tag SFs for btag inputs

        ####################
        #  Fill histogram  #
        ####################
        for syst in systematics:
            if self.isSyst == False and syst != "nominal":
                break
            if self.noHist:
                break
            weight = (
                weights.weight()
                if syst == "nominal" or syst == shift_name
                else weights.weight(modifier=syst)
            )  # shift up/down for weight systematics

            # fill the histogram (check axis defintion in histogrammer and following the order)
            # TODO:
            # - Jet pt, eta
            # - QGL variables
            # - PNet
            # - DeepJet
            # - ParT
            output["jet_pt"].fill(
                syst,
                flatten(genflavor[:, 0]),
                flatten(sjets[:, 0].pt),
                weight=weight,
            )
            output["mu_pt"].fill(syst, flatten(smu[:, 0].pt), weight=weight)
            output["dr_mujet"].fill(
                syst,
                flatten(genflavor[:, 0]),  # the fill content should always flat arrays
                flatten(sjets[:, 0].delta_r(smu[:, 0])),
                weight=weight,
            )
        #######################
        #  Create root files  # : Save arrays in to root file, keep axis structure
        #######################
        if self.isArray:
            # Keep the structure of events and pruned the object size
            pruned_ev = events[event_level]  # pruned events

            pruned_ev.Muon = smu  # replace muon collections with selected muon
        if self.isArray:
            # Keep the structure of events and pruned the object size
            pruned_ev = events[event_level]
            pruned_ev["SelJet"] = sjets
            pruned_ev["Muon"] = smu

            # Add custom variables
            if not isRealData:
                pruned_ev["weight"] = weights.weight()
                for ind_wei in weights.weightStatistics.keys():
                    pruned_ev[f"{ind_wei}_weight"] = weights.partial_weight(
                        include=[ind_wei]
                    )
            array_writer(self, pruned_ev, events, systematics[0], dataset, isRealData)

        return {dataset: output}

    ## post process, return the accumulator, compressed
    def postprocess(self, accumulator):
        return accumulator