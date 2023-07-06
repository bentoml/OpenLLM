(ns openllm.components.side-bar.subs
  (:require [openllm.db :as db]
            [openllm.subs :as root-subs]
            [re-frame.core :refer [reg-sub]]))

(def parameter-id->human-readable
  {::db/temperature "Temperature"
   ::db/top_k "Top K"
   ::db/top_p "Top P"
   ::db/typical_p "Typical P"
   ::db/epsilon_cutoff "Epsilon Cutoff"
   ::db/eta_cutoff "Eta Cutoff"
   ::db/diversity_penalty "Diversity Penalty"
   ::db/repetition_penalty "Repetition Penalty"
   ::db/encoder_repetition_penalty "Encoder Repetition Penalty"
   ::db/length_penalty "Length Penalty"
   ::db/max_new_tokens "Maximum New Tokens"
   ::db/min_length "Minimum Length"
   ::db/min_new_tokens "Minimum New Tokens"
   ::db/early_stopping "Early Stopping"
   ::db/max_time "Maximum Time"
   ::db/num_beams "Number of Beams"
   ::db/num_beam_groups "Number of Beam Groups"
   ::db/penalty_alpha "Penalty Alpha"
   ::db/use_cache "Use Cache"})

(reg-sub
 ::side-bar-open?
 (fn [db _]
   (:side-bar-open? db)))

(reg-sub
 ::human-readable-config
 :<- [::root-subs/model-config]
 (fn [model-config _]
   (map (fn [[k v]]
          [k {:value v :name (parameter-id->human-readable k)}])
        model-config)))
