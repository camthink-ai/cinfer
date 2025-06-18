import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';

interface GlobalStore {
    /**
     * infer Engines
     */
    inferEngines?: string[];

    /**
     * Update infer Engines
     */
    setInferEngines: (engines?: string[]) => void;
}

const useGlobalStore = create(
    immer<GlobalStore>(set => ({
        inferEngines: [],

        setInferEngines: engines => {
            set(state => {
                state.inferEngines = engines;
            });
        },
    })),
);

export default useGlobalStore;
