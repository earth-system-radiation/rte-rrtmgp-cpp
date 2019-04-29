#ifndef GAS_CONCS_H
#define GAS_CONCS_H

#include <vector>

template<typename TF>
class Gas_concs
{
    public:
        // Insert new gas into the map.
        void set_vmr(const std::string& name, const Array<TF,2>& data)
        {
            gas_concs_map.emplace(name, data);
        }

        // Check if gas exists in map.
        bool exists(const std::string& name) const { return gas_concs_map.count(name) != 0; }

    private:
        std::map<std::string, Array<TF,2>> gas_concs_map;
};
#endif
