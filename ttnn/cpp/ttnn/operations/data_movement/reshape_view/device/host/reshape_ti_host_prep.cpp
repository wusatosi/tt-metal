#include <math.h>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape_common.hpp"

namespace ttnn::operations::data_movement::tile_reshape{

operation::ProgramWithCallbacks tile_reshape_preparer(const Tensor& input, Tensor& output, )
{

}
}; // namespace ttnn::operations::data_movement::rm_reshape
