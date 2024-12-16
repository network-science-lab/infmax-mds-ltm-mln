from typing import Any
import network_diffusion as nd


class MDSError(BaseException):
    """Exception raised where MDS is smaller than the budget size."""


class MDSLimitedMLTModel(nd.models.MLTModel):
    """
    Slightly modified Multilayer Linear Threshold Model.

    It raises ValueError if provided ranking list to select seeds is shorter than the number of
    actors. That is to detect if it's sensible to use MDS in seed selection process.
    """

    def determine_initial_states(self, net: nd.mln.MultilayerNetwork) -> list[nd.models.NetworkUpdateBuffer]:
        budget = self._compartmental_graph.get_seeding_budget_for_network(net=net, actorwise=True)
        preselected_actors = self._seed_selector.preselected_actors

        # raise flag if using MDS is not wise
        if budget[self.PROCESS_NAME][self.ACTIVE_STATE] > len(preselected_actors):
            raise MDSError("Budget is bigger than lenght of the ranking!")

        # a default case, exactly like in network_diffusion
        if len(preselected_actors) == net.get_actors_num():
            return self.set_states(preselected_actors, budget)
        
        # otherwise construct ranking list as in network_diffusion and process it normally
        ranking_list = [*preselected_actors, *set(net.get_actors()).difference(set(preselected_actors))]
        return self.set_states(ranking_list, budget)

    def set_states(
        self, ranking: list[nd.mln.MLNetworkActor], budget: dict[str, dict[Any, int]]
    ) -> list[nd.models.NetworkUpdateBuffer]:
        states = []
        for idx, actor in enumerate(ranking):
            if idx < budget[self.PROCESS_NAME][self.ACTIVE_STATE]:
                a_state = self.ACTIVE_STATE
            else:
                a_state = self.INACTIVE_STATE
            for l_name in actor.layers:
                states.append(nd.models.NetworkUpdateBuffer(actor.actor_id, l_name, a_state))
        return states
