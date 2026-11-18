/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_LAYOUT_HPP
#define TLA_LAYOUT_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/numeric_size.hpp"
#include "tla/integral/integral_constant.hpp"
#include "tla/integral/integral_math.hpp"
#include "tla/utils/math.hpp"
#include "tla/tuple/tuple.hpp"
#include "tla/tuple/tuple_algorithms.hpp"
#include "tla/tuple/tuple_math.hpp"
#include "tla/utils/functional.hpp"
#include "tla/stride.hpp"
#include "catlass/layout/layout.hpp"

namespace tla {

// Type aliases and factory functions
template <class... Shapes>
using Shape = tla::tuple<Shapes...>;
template <class... Ts>
CATLASS_HOST_DEVICE constexpr Shape<Ts...> make_shape(Ts const&... t)
{
    return {t...};
}
// PascalCase alias
template <class... Ts>
CATLASS_HOST_DEVICE constexpr Shape<Ts...> MakeShape(Ts const&... t)
{
    return make_shape(t...);
}

template <class... Strides>
using Stride = tla::tuple<Strides...>;
template <class... Ts>
CATLASS_HOST_DEVICE constexpr Stride<Ts...> make_stride(Ts const&... t)
{
    return {t...};
}
// PascalCase alias
template <class... Ts>
CATLASS_HOST_DEVICE constexpr Stride<Ts...> MakeStride(Ts const&... t)
{
    return make_stride(t...);
}

template <class... Coords>
using Coord = tla::tuple<Coords...>;
template <class... Ts>
CATLASS_HOST_DEVICE constexpr Coord<Ts...> make_coord(Ts const&... t)
{
    return {t...};
}
// PascalCase alias
template <class... Ts>
CATLASS_HOST_DEVICE constexpr Coord<Ts...> MakeCoord(Ts const&... t)
{
    return make_coord(t...);
}

template <class... Strides>
using Step = tla::tuple<Strides...>;
template <class... Ts>
CATLASS_HOST_DEVICE constexpr Step<Ts...> make_step(Ts const&... t)
{
    return {t...};
}

template <class... Layouts>
using Tile = tla::tuple<Layouts...>;
template <class... Ts>
CATLASS_HOST_DEVICE constexpr Tile<Ts...> make_tile(Ts const&... t)
{
    return {t...};
}

// Layout core type
namespace detail {

template <class Shape>
CATLASS_HOST_DEVICE constexpr auto make_origin_shape(Shape const& shape)
{
    return conditional_return(is_tuple<Shape>{}, product_each(shape), shape);
}

} // namespace detail

template <class Shape, class Stride, class OriginShape = decltype(detail::make_origin_shape(tla::declval<Shape>()))>
struct Layout : private tla::tuple<Shape, Stride, OriginShape> {
    CATLASS_HOST_DEVICE constexpr Layout(
        Shape const& shape = {}, Stride const& stride = {}, OriginShape const& originShape = {})
        : tla::tuple<Shape, Stride, OriginShape>(shape, stride, originShape)
    {}

    static constexpr int rank = rank_v<Stride>;
    static constexpr int depth = depth_v<Stride>;

    template <int... I>
    CATLASS_HOST_DEVICE constexpr decltype(auto) shape()
    {
        return get<0, I...>(static_cast<tla::tuple<Shape, Stride, OriginShape>&>(*this));
    }

    template <int... I>
    CATLASS_HOST_DEVICE constexpr decltype(auto) shape() const
    {
        return get<0, I...>(static_cast<tla::tuple<Shape, Stride, OriginShape> const&>(*this));
    }

    template <int... I>
    CATLASS_HOST_DEVICE constexpr decltype(auto) stride()
    {
        return get<1, I...>(static_cast<tla::tuple<Shape, Stride, OriginShape>&>(*this));
    }

    template <int... I>
    CATLASS_HOST_DEVICE constexpr decltype(auto) stride() const
    {
        return get<1, I...>(static_cast<tla::tuple<Shape, Stride, OriginShape> const&>(*this));
    }

    template <int... I>
    CATLASS_HOST_DEVICE constexpr decltype(auto) originShape()
    {
        return get<2, I...>(static_cast<tla::tuple<Shape, Stride, OriginShape>&>(*this));
    }

    template <int... I>
    CATLASS_HOST_DEVICE constexpr decltype(auto) originShape() const
    {
        return get<2, I...>(static_cast<tla::tuple<Shape, Stride, OriginShape> const&>(*this));
    }

    template <class Coord>
    CATLASS_HOST_DEVICE constexpr auto operator()(Coord const& coord) const
    {
        return crd2offset(coord, shape(), stride());
    }
};

template <class Layout>
struct is_layout : false_type {};
template <class Shape, class Stride, class OriginShape>
struct is_layout<Layout<Shape, Stride, OriginShape>> : true_type {};
template <class T>
inline constexpr bool is_layout_v = is_layout<T>::value;

// Layout construction
template <
    class Shape, class Stride, class OriginShape,
    TLA_REQUIRES((is_int_tuple_v<Shape>) && (is_int_tuple_v<Stride>) && (is_int_tuple_v<OriginShape>))>
CATLASS_HOST_DEVICE constexpr auto make_layout(Shape const& shape, Stride const& stride, OriginShape const& originShape)
{
    return Layout<Shape, Stride, OriginShape>(shape, stride, originShape);
}

// PascalCase alias
template <class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr auto MakeLayout(Shape const& shape, Stride const& stride, OriginShape const& originShape)
{
    return make_layout(shape, stride, originShape);
}

template <class Shape, class Stride, TLA_REQUIRES((is_int_tuple_v<Shape>) && (is_int_tuple_v<Stride>))>
CATLASS_HOST_DEVICE constexpr auto make_layout(Shape const& shape, Stride const& stride)
{
    const auto originShape = detail::make_origin_shape(shape);
    return make_layout(shape, stride, originShape);
}

// PascalCase alias
template <class Shape, class Stride>
CATLASS_HOST_DEVICE constexpr auto MakeLayout(Shape const& shape, Stride const& stride)
{
    return make_layout(shape, stride);
}

template <class Shape, TLA_REQUIRES(is_int_tuple_v<Shape>)>
CATLASS_HOST_DEVICE constexpr auto make_layout(Shape const& shape)
{
    const auto stride = compact_col_major(shape);
    return make_layout(shape, stride);
}

template <class Layout0, class... Layouts, TLA_REQUIRES((is_layout_v<Layout0>) && (is_layout_v<Layouts> && ...))>
CATLASS_HOST_DEVICE constexpr auto make_layout(Layout0 const& layout0, Layouts const&... layouts)
{
    return make_layout(
        make_shape(layout0.shape(), layouts.shape()...), make_stride(layout0.stride(), layouts.stride()...),
        make_shape(layout0.originShape(), layouts.originShape()...));
}

template <class Shape, class Order>
CATLASS_HOST_DEVICE constexpr auto make_ordered_layout(Shape const& shape, Order const& order)
{
    return make_layout(shape, compact_order(shape, order));
}

template <class Shape, class Order, class OriginShape>
CATLASS_HOST_DEVICE constexpr auto make_ordered_layout(
    Shape const& shape, Order const& order, OriginShape const& originShape)
{
    return make_layout(shape, compact_order(shape, order), originShape);
}

// Layout accessors (free functions)
template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr decltype(auto) shape(Layout<Shape, Stride, OriginShape>& layout)
{
    return layout.template shape<Is...>();
}

template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr decltype(auto) shape(Layout<Shape, Stride, OriginShape> const& layout)
{
    return layout.template shape<Is...>();
}

template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr decltype(auto) stride(Layout<Shape, Stride, OriginShape>& layout)
{
    return layout.template stride<Is...>();
}

template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr decltype(auto) stride(Layout<Shape, Stride, OriginShape> const& layout)
{
    return layout.template stride<Is...>();
}

template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr decltype(auto) originShape(Layout<Shape, Stride, OriginShape>& layout)
{
    return layout.template originShape<Is...>();
}

template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr decltype(auto) originShape(Layout<Shape, Stride, OriginShape> const& layout)
{
    return layout.template originShape<Is...>();
}

template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr auto rank(Layout<Shape, Stride, OriginShape> const& layout)
{
    return rank(shape<Is...>(layout));
}

template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr auto depth(Layout<Shape, Stride, OriginShape> const& layout)
{
    return depth(shape<Is...>(layout));
}

// Layout manipulation (like a tuple of pairs)

// get<Is...>(layout): extract sublayout at recursive indices Is...
// For Is... = <I0,I1,...,IN>, equivalent to get<IN>(...get<I1>(get<I0>(layout)))
template <size_t... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr auto get(Layout<Shape, Stride, OriginShape> const& layout)
{
    return make_layout(get<Is...>(shape(layout)), get<Is...>(stride(layout)), get<Is...>(originShape(layout)));
}

// take<B, E>(layout): extract sublayout with modes in range [B, E)
template <int B, int E, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr auto take(Layout<Shape, Stride, OriginShape> const& layout)
{
    return make_layout(take<B, E>(shape(layout)), take<B, E>(stride(layout)), take<B, E>(originShape(layout)));
}

// select<Is...>(layout): extract sublayout with modes at indices Is...
template <size_t... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr auto select(Layout<Shape, Stride, OriginShape> const& layout)
{
    return make_layout(select<Is...>(shape(layout)), select<Is...>(stride(layout)), select<Is...>(originShape(layout)));
}

// Codomain operations

// coshape = (shape - 1) * |stride| + 1 (per mode, recursive)
template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr auto coshape(Layout<Shape, Stride, OriginShape> const& layout)
{
    auto m1_shapes = transform_leaf([](auto s) { return s - Int<1>{}; }, shape<Is...>(layout));
    auto abs_strides = transform_leaf(abs_fn{}, stride<Is...>(layout));
    auto co_coord = inner_product(m1_shapes, abs_strides);
    return transform_leaf([](auto c) { return c + Int<1>{}; }, co_coord);
}

template <int... Is, class Shape, class Stride, class OriginShape>
CATLASS_HOST_DEVICE constexpr auto cosize(Layout<Shape, Stride, OriginShape> const& layout)
{
    return size(coshape<Is...>(layout));
}

template <class Layout_>
using cosize_t = decltype(cosize(std::declval<Layout_>()));
// Tag-dispatched layout construction

namespace detail {

template <class Element>
struct element_alignment {
    static constexpr auto value = max(Catlass::BytesToElements<Element>(_1{}), _1{});
};

template <>
struct element_alignment<void> {
    static constexpr auto value = _1{};
};

template <class Element>
inline constexpr auto element_alignment_v = element_alignment<Element>::value;

template <class LayoutTag>
struct FractalSpec {};

template <>
struct FractalSpec<Catlass::layout::zN> {
    using Order = tla::Step<tla::Step<tla::Int<1>, tla::Int<2>>, tla::Step<tla::Int<0>, tla::Int<3>>>;
    template <class Element>
    static constexpr auto fractal = tla::Shape<
        tla::Int<Catlass::C0_NUM_PER_FRACTAL>, tla::Int<Catlass::BytesToElements<Element>(Catlass::BYTE_PER_C0)>>{};
};

template <>
struct FractalSpec<Catlass::layout::nZ> {
    using Order = tla::Step<tla::Step<tla::Int<0>, tla::Int<3>>, tla::Step<tla::Int<1>, tla::Int<2>>>;
    template <class Element>
    static constexpr auto fractal = tla::Shape<
        tla::Int<Catlass::BytesToElements<Element>(Catlass::BYTE_PER_C0)>, tla::Int<Catlass::C0_NUM_PER_FRACTAL>>{};
};

template <>
struct FractalSpec<Catlass::layout::zZ> {
    using Order = tla::Step<tla::Step<tla::Int<1>, tla::Int<3>>, tla::Step<tla::Int<0>, tla::Int<2>>>;
    template <class Element>
    static constexpr auto fractal = tla::Shape<
        tla::Int<Catlass::C0_NUM_PER_FRACTAL>, tla::Int<Catlass::BytesToElements<Element>(Catlass::BYTE_PER_C0)>>{};
};

template <>
struct FractalSpec<Catlass::layout::nN> {
    using Order = tla::Step<tla::Step<tla::Int<0>, tla::Int<2>>, tla::Step<tla::Int<1>, tla::Int<3>>>;
    template <class Element>
    static constexpr auto fractal = tla::Shape<
        tla::Int<Catlass::BytesToElements<Element>(Catlass::BYTE_PER_C0)>, tla::Int<Catlass::C0_NUM_PER_FRACTAL>>{};
};

template <>
struct FractalSpec<Catlass::layout::L0C> {
    using Order = tla::Step<tla::Step<tla::Int<1>, tla::Int<2>>, tla::Step<tla::Int<0>, tla::Int<3>>>;
    template <class Element>
    static constexpr auto fractal =
        tla::Shape<tla::Int<Catlass::C0_NUM_PER_FRACTAL>, tla::Int<Catlass::C0_NUM_PER_FRACTAL>>{};
};

template <class LayoutTag, class = void>
struct is_fractal_layout : false_type {};

template <class LayoutTag>
struct is_fractal_layout<LayoutTag, std::void_t<typename FractalSpec<LayoutTag>::Order>> : true_type {};

} // end namespace detail

template <class Shape, class Fractal>
CATLASS_HOST_DEVICE constexpr auto make_fractal_shape(Shape const& shape, Fractal const& fractal)
{
    return detail::transform_impl(
        [](auto const& s, auto const& f) { return make_tuple(f, ceil_div(s, f)); }, shape, fractal);
}

template <class LayoutTag, class Element = void, class Shape>
CATLASS_HOST_DEVICE constexpr auto make_layout(Shape const& shape)
{
    static_assert(is_int_tuple_v<Shape>, "make_layout<LayoutTag>(shape): shape must be an integral or int tuple");
    if constexpr (is_same_v<LayoutTag, Catlass::layout::VectorLayout>) {
        static_assert(rank_v<Shape> == 1, "VectorLayout requires rank-1 shape");
        return make_layout(wrap(shape), make_stride(Int<1>{}));
    } else if constexpr (is_same_v<LayoutTag, Catlass::layout::RowMajor>) {
        static_assert(rank_v<Shape> == 2, "RowMajor requires rank-2 shape");
        const auto stride = round_up(compact_row_major(shape), make_stride(detail::element_alignment_v<Element>, _1{}));
        return make_layout(shape, stride);
    } else if constexpr (is_same_v<LayoutTag, Catlass::layout::ColumnMajor>) {
        static_assert(rank_v<Shape> == 2, "ColumnMajor requires rank-2 shape");
        const auto stride = round_up(compact_col_major(shape), make_stride(_1{}, detail::element_alignment_v<Element>));
        return make_layout(shape, stride);
    } else if constexpr (detail::is_fractal_layout<LayoutTag>::value) {
        static_assert(rank_v<Shape> == 2, "fractal layout requires rank-2 shape");
        const auto fractalShape = make_fractal_shape(shape, detail::FractalSpec<LayoutTag>::template fractal<Element>);
        return make_ordered_layout(fractalShape, typename detail::FractalSpec<LayoutTag>::Order{}, shape);
    } else {
        static_assert(dependent_false<LayoutTag>, "Unsupported LayoutTag for make_layout.");
    }
}

template <class LayoutTag, class Element = void, class InnerShape, class OuterShape>
CATLASS_HOST_DEVICE constexpr auto make_layout(InnerShape const& innerShape, OuterShape const& outerShape)
{
    static_assert(
        is_int_tuple_v<InnerShape> && is_int_tuple_v<OuterShape>,
        "make_layout<LayoutTag>(inner, outer): both shapes must be integral or int tuples");
    const auto innerLayout = make_layout<LayoutTag, Element>(innerShape);
    const auto outerLayout = make_layout(outerShape, compact_col_major(outerShape, cosize(innerLayout)));
    return make_layout(
        tuple_cat(innerLayout.shape(), wrap(outerLayout.shape())),
        tuple_cat(innerLayout.stride(), wrap(outerLayout.stride())),
        tuple_cat(innerLayout.originShape(), wrap(outerLayout.originShape())));
}

template <class LayoutTag>
CATLASS_HOST_DEVICE constexpr auto MakeLayoutFromTag(LayoutTag const& tag)
{
    static_assert(
        is_same_v<LayoutTag, Catlass::layout::RowMajor> || is_same_v<LayoutTag, Catlass::layout::ColumnMajor> ||
            is_same_v<LayoutTag, Catlass::layout::VectorLayout> || is_same_v<LayoutTag, Catlass::layout::zN> ||
            is_same_v<LayoutTag, Catlass::layout::nZ> || is_same_v<LayoutTag, Catlass::layout::L0C>,
        "Unsupported LayoutTag for MakeLayoutFromTag, only support Catlass::layout::RowMajor or"
        "Catlass::layout::ColumnMajor or Catlass::layout::VectorLayout or Catlass::layout::zN or Catlass::layout::nZ "
        "or Catlass::layout::L0C");

    if constexpr (is_same_v<LayoutTag, Catlass::layout::VectorLayout>) {
        return MakeLayout(MakeShape(tag.shape(0)), MakeStride(tag.stride(0)), MakeShape(tag.shape(0)));
    } else if constexpr (is_same_v<LayoutTag, Catlass::layout::RowMajor>) {
        return MakeLayout(
            MakeShape(tag.shape(0), tag.shape(1)), MakeStride(tag.stride(0), Int<1>{}),
            MakeShape(tag.shape(0), tag.shape(1)));
    } else if constexpr (is_same_v<LayoutTag, Catlass::layout::ColumnMajor>) {
        return MakeLayout(
            MakeShape(tag.shape(0), tag.shape(1)), MakeStride(Int<1>{}, tag.stride(1)),
            MakeShape(tag.shape(0), tag.shape(1)));
    } else { // zN or nZ or L0C
        return MakeLayout(
            MakeShape(MakeShape(tag.shape(0), tag.shape(1)), MakeShape(tag.shape(2), tag.shape(3))),
            MakeStride(MakeStride(tag.stride(0), tag.stride(1)), MakeStride(tag.stride(2), tag.stride(3))),
            MakeShape(tag.orgShape(0), tag.orgShape(1)));
    }
}

template <class T>
CATLASS_HOST_DEVICE constexpr auto MakeLayout(T const& len)
{
    return MakeLayout(MakeShape(len), MakeStride(Int<1>{}), MakeShape(len));
}

template <class Element, class LayoutTag, class T, class U>
CATLASS_HOST_DEVICE constexpr auto MakeLayout(T const& rows, U const& cols)
{
    static_assert(
        is_same_v<LayoutTag, Catlass::layout::RowMajor> || is_same_v<LayoutTag, Catlass::layout::ColumnMajor> ||
            is_same_v<LayoutTag, Catlass::layout::VectorLayout> || is_same_v<LayoutTag, Catlass::layout::zN> ||
            is_same_v<LayoutTag, Catlass::layout::nZ> || is_same_v<LayoutTag, Catlass::layout::zZ> ||
            is_same_v<LayoutTag, Catlass::layout::L0C> || is_same_v<LayoutTag, Catlass::layout::Weight4BitnZ>,
        "Unsupported LayoutTag for MakeLayoutFromTag, only support Catlass::layout::RowMajor or"
        "Catlass::layout::ColumnMajor or Catlass::layout::zN or Catlass::layout::nZ or Catlass::layout::zZ or "
        "Catlass::layout::L0C or Catlass::layout::Weight4BitnZ");

    constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BytesToElements<Element>(Catlass::BYTE_PER_C0);
    constexpr uint32_t ELE_NUM_PER_FRACTAL =
        Catlass::BytesToBits(Catlass::BYTE_PER_FRACTAL) / Catlass::SizeOfBits<Element>::value;

    if constexpr (is_same_v<LayoutTag, Catlass::layout::VectorLayout>) {
        return MakeLayout(MakeShape(cols), MakeStride(Int<1>{}), MakeShape(cols));
    } else if constexpr (is_same_v<LayoutTag, Catlass::layout::RowMajor>) {
#if (defined(CATLASS_ARCH) && (CATLASS_ARCH == 3510 || CATLASS_ARCH == 9201 || CATLASS_ARCH == 9202))
        if constexpr (is_same_v<Element, float4_e2m1x2_t> || is_same_v<Element, float4_e1m2x2_t>) {
            return MakeLayout(
                MakeShape(rows, cols), MakeStride((int64_t)RoundUp(cols, 2), Int<1>{}), MakeShape(rows, cols));
        }
#endif
        return MakeLayout(MakeShape(rows, cols), MakeStride((int64_t)cols, Int<1>{}), MakeShape(rows, cols));
    } else if constexpr (is_same_v<LayoutTag, Catlass::layout::ColumnMajor>) {
#if (defined(CATLASS_ARCH) && (CATLASS_ARCH == 3510 || CATLASS_ARCH == 9201 || CATLASS_ARCH == 9202))
        if constexpr (is_same_v<Element, float4_e2m1x2_t> || is_same_v<Element, float4_e1m2x2_t>) {
            return MakeLayout(
                MakeShape(rows, cols), MakeStride(Int<1>{}, (int64_t)RoundUp(rows, 2)), MakeShape(rows, cols));
        }
#endif
        return MakeLayout(MakeShape(rows, cols), MakeStride(Int<1>{}, (int64_t)rows), MakeShape(rows, cols));
    } else if constexpr (is_same_v<LayoutTag, Catlass::layout::zN>) {
        return MakeLayout(
            MakeShape(
                MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(rows, Int<Catlass::C0_NUM_PER_FRACTAL>{})),
                MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
            MakeStride(
                MakeStride(Int<ELE_NUM_PER_C0>{}, Int<ELE_NUM_PER_FRACTAL>{}),
                MakeStride(Int<1>{}, RoundUp((int64_t)rows, Int<Catlass::C0_NUM_PER_FRACTAL>{}) * ELE_NUM_PER_C0)),
            MakeShape(rows, cols));
    } else if constexpr (is_same_v<LayoutTag, Catlass::layout::zZ>) {
        return MakeLayout(
            MakeShape(
                MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(rows, Int<Catlass::C0_NUM_PER_FRACTAL>{})),
                MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
            MakeStride(
                MakeStride(
                    Int<ELE_NUM_PER_C0>{}, RoundUp((int64_t)cols, Int<ELE_NUM_PER_C0>{}) * Catlass::C0_NUM_PER_FRACTAL),
                MakeStride(Int<1>{}, Int<ELE_NUM_PER_FRACTAL>{})),
            MakeShape(rows, cols));
    } else if constexpr (is_same_v<LayoutTag, Catlass::layout::L0C>) {
        constexpr uint32_t ELE_NUM_PER_FRACTAL = 256;
        return MakeLayout(
            MakeShape(
                MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(rows, Int<Catlass::C0_NUM_PER_FRACTAL>{})),
                MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(cols, Int<Catlass::C0_NUM_PER_FRACTAL>{}))),
            MakeStride(
                MakeStride(Int<Catlass::C0_NUM_PER_FRACTAL>{}, Int<ELE_NUM_PER_FRACTAL>{}),
                MakeStride(
                    Int<1>{},
                    RoundUp((int64_t)rows, Int<Catlass::C0_NUM_PER_FRACTAL>{}) * Catlass::C0_NUM_PER_FRACTAL)),
            MakeShape(rows, cols));
    } else if constexpr (is_same_v<LayoutTag, Catlass::layout::Weight4BitnZ>) {
        constexpr uint32_t ELE_NUM_PER_C0 = 32;
        constexpr uint32_t ELE_NUM_PER_FRACTAL = 512;
        return MakeLayout(
            MakeShape(
                MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(rows, Int<ELE_NUM_PER_C0>{})),
                MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(cols, Int<Catlass::C0_NUM_PER_FRACTAL>{}))),
            MakeStride(
                MakeStride(Int<1>{}, RoundUp((int64_t)cols, Int<Catlass::C0_NUM_PER_FRACTAL>{}) * ELE_NUM_PER_C0),
                MakeStride(Int<ELE_NUM_PER_C0>{}, Int<ELE_NUM_PER_FRACTAL>{})),
            MakeShape(rows, cols));
    } else {
        return MakeLayout(
            MakeShape(
                MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(rows, Int<ELE_NUM_PER_C0>{})),
                MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(cols, Int<Catlass::C0_NUM_PER_FRACTAL>{}))),
            MakeStride(
                MakeStride(Int<1>{}, RoundUp((int64_t)cols, Int<Catlass::C0_NUM_PER_FRACTAL>{}) * ELE_NUM_PER_C0),
                MakeStride(Int<ELE_NUM_PER_C0>{}, Int<ELE_NUM_PER_FRACTAL>{})),
            MakeShape(rows, cols));
    }
}

#if (defined(CATLASS_ARCH) && (CATLASS_ARCH == 3510 || CATLASS_ARCH == 9201 || CATLASS_ARCH == 9202))
template <class Element, class LayoutTag, bool isMxScaleB, class T, class U>
CATLASS_HOST_DEVICE constexpr auto MakeMxScaleLayout(T const& rows, U const& cols)
{
    static_assert(
        is_same_v<Element, float8_e8m0_t> &&
            (is_same_v<LayoutTag, Catlass::layout::RowMajor> || is_same_v<LayoutTag, Catlass::layout::ColumnMajor> ||
             is_same_v<LayoutTag, Catlass::layout::zZ> || is_same_v<LayoutTag, Catlass::layout::nN>),
        "only support RowMajor, ColumnMajor, zZ, nN in fp8_e8m0_t dtype");

    constexpr uint32_t ELE_NUM_PER_C0 = 2;
    constexpr uint32_t ELE_NUM_PER_FRACTAL = 32;

    if constexpr (is_same_v<LayoutTag, Catlass::layout::RowMajor>) {
        if constexpr (!isMxScaleB) {
            return MakeLayout(
                MakeShape(rows, MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
                MakeStride(RoundUp(cols, Int<ELE_NUM_PER_C0>{}), MakeStride(Int<1>{}, Int<ELE_NUM_PER_C0>{})),
                MakeShape(rows, cols));
        } else {
            return MakeLayout(
                MakeShape(MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(rows, Int<ELE_NUM_PER_C0>{})), cols),
                MakeStride(MakeStride(Int<1>{}, cols * ELE_NUM_PER_C0), Int<ELE_NUM_PER_C0>{}), MakeShape(rows, cols));
        }
    } else if constexpr (is_same_v<LayoutTag, Catlass::layout::ColumnMajor>) {
        if constexpr (!isMxScaleB) {
            return MakeLayout(
                MakeShape(rows, MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
                MakeStride(Int<ELE_NUM_PER_C0>{}, MakeStride(Int<1>{}, rows * ELE_NUM_PER_C0)), MakeShape(rows, cols));
        } else {
            return MakeLayout(
                MakeShape(MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(rows, Int<ELE_NUM_PER_C0>{})), cols),
                MakeStride(MakeStride(Int<1>{}, Int<ELE_NUM_PER_C0>{}), RoundUp(rows, Int<ELE_NUM_PER_C0>{})),
                MakeShape(rows, cols));
        }
    } else if constexpr (is_same_v<LayoutTag, Catlass::layout::zZ>) {
        return MakeLayout(
            MakeShape(
                MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(rows, Int<Catlass::C0_NUM_PER_FRACTAL>{})),
                MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
            MakeStride(
                MakeStride(
                    Int<ELE_NUM_PER_C0>{}, RoundUp((int64_t)cols, Int<ELE_NUM_PER_C0>{}) * Catlass::C0_NUM_PER_FRACTAL),
                MakeStride(Int<1>{}, Int<ELE_NUM_PER_FRACTAL>{})),
            MakeShape(rows, cols));
    } else {
        return MakeLayout(
            MakeShape(
                MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(rows, Int<ELE_NUM_PER_C0>{})),
                MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(cols, Int<Catlass::C0_NUM_PER_FRACTAL>{}))),
            MakeStride(
                MakeStride(Int<1>{}, Int<ELE_NUM_PER_FRACTAL>{}),
                MakeStride(
                    Int<ELE_NUM_PER_C0>{},
                    RoundUp((int64_t)rows, Int<ELE_NUM_PER_C0>{}) * Catlass::C0_NUM_PER_FRACTAL)),
            MakeShape(rows, cols));
    }
}
#endif

// Specialized layout factories
template <class T, class U>
CATLASS_HOST_DEVICE constexpr auto MakeLayoutL0C(T const& rows, U const& cols)
{
    constexpr uint32_t ELE_NUM_PER_FRACTAL = 256;
    return MakeLayout(
        MakeShape(
            MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(rows, Int<Catlass::C0_NUM_PER_FRACTAL>{})),
            MakeShape(Int<Catlass::C0_NUM_PER_FRACTAL>{}, CeilDiv(cols, Int<Catlass::C0_NUM_PER_FRACTAL>{}))),
        MakeStride(
            MakeStride(Int<Catlass::C0_NUM_PER_FRACTAL>{}, Int<ELE_NUM_PER_FRACTAL>{}),
            MakeStride(
                Int<1>{}, RoundUp((int64_t)rows, Int<Catlass::C0_NUM_PER_FRACTAL>{}) * Catlass::C0_NUM_PER_FRACTAL)),
        MakeShape(rows, cols));
}

template <class Element, class T1, class T2, class T3, class T4>
CATLASS_HOST_DEVICE constexpr auto MakeLayoutFmap(T1 const& batch, T2 const& cin1, T3 const& h, T4 const& w)
{
    constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    const int64_t strideH = w * ELE_NUM_PER_C0;
    const int64_t strideCin1 = h * strideH;
    const int64_t strideBatch = cin1 * strideCin1;
    return MakeLayout(
        MakeShape(
            static_cast<uint32_t>(batch), static_cast<uint32_t>(cin1), static_cast<uint32_t>(h),
            static_cast<uint32_t>(w), Int<ELE_NUM_PER_C0>{}),
        MakeStride(strideBatch, strideCin1, strideH, Int<ELE_NUM_PER_C0>{}, Int<1>{}),
        MakeShape(
            static_cast<uint32_t>(batch), static_cast<uint32_t>(cin1), static_cast<uint32_t>(h),
            static_cast<uint32_t>(w), ELE_NUM_PER_C0));
}

template <class Element, class PositionType, class T1, class T2, class T3, class T4>
CATLASS_HOST_DEVICE constexpr auto MakeLayoutFilter(T1 const& cin1, T2 const& kh, T3 const& kw, T4 const& cout)
{
    constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BYTE_PER_C0 / sizeof(Element);
    const uint32_t coutRound =
        is_same_v<PositionType, Catlass::Arch::PositionL1> ? RoundUp(cout, Catlass::C0_NUM_PER_FRACTAL) : cout;
    const int64_t strideKw = coutRound * ELE_NUM_PER_C0;
    const int64_t strideKh = kw * strideKw;
    const int64_t strideCin1 = kh * strideKh;
    return MakeLayout(
        MakeShape(
            static_cast<uint32_t>(cin1), static_cast<uint32_t>(kh), static_cast<uint32_t>(kw),
            static_cast<uint32_t>(coutRound), Int<ELE_NUM_PER_C0>{}),
        MakeStride(strideCin1, strideKh, strideKw, Int<ELE_NUM_PER_C0>{}, Int<1>{}),
        MakeShape(
            static_cast<uint32_t>(cin1), static_cast<uint32_t>(kh), static_cast<uint32_t>(kw),
            static_cast<uint32_t>(coutRound), ELE_NUM_PER_C0));
}

// Layout type predicates
namespace detail {

template <class Layout, class Enable = void>
struct isVector {
    static bool const value = false;
};

template <class Layout>
struct isVector<Layout, std::enable_if_t<Layout::depth == 1 && Layout::rank == 1>> {
    static bool const value = (stride<0>(Layout{}) == 1);
};

template <class Layout, class Enable = void>
struct isRowMajor {
    static bool const value = false;
};

template <class Layout>
struct isRowMajor<Layout, std::enable_if_t<Layout::depth == 1 && Layout::rank == 2>> {
    static bool const value = (stride<1>(Layout{}) == 1);
};

template <class Layout, class Enable = void>
struct isColumnMajor {
    static bool const value = false;
};

template <class Layout>
struct isColumnMajor<Layout, std::enable_if_t<Layout::depth == 1 && Layout::rank == 2>> {
    static bool const value = (stride<0>(Layout{}) == 1);
};

template <class Element, class Layout, class Enable1 = void, class Enable2 = void>
struct iszN {
    static bool const value = false;
};

template <class Element, class Layout>
struct iszN<
    Element, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<
        rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2 &&
        rank_v<decltype(stride<0>(Layout{}))> == 2 && rank_v<decltype(stride<1>(Layout{}))> == 2>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BytesToElements<Element>(Catlass::BYTE_PER_C0);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL =
        Catlass::BytesToBits(Catlass::BYTE_PER_FRACTAL) / Catlass::SizeOfBits<Element>::value;
    static bool const value =
        (shape<0, 0>(Layout{}) == Catlass::C0_NUM_PER_FRACTAL && shape<1, 0>(Layout{}) == ELE_NUM_PER_C0 &&
         stride<1, 0>(Layout{}) == 1 && stride<0, 1>(Layout{}) == ELE_NUM_PER_FRACTAL);
};

/*
For matmul m axis is not c0 Aligned.
Exp: oriShape(m, k) : (127, 256)
zNUnAlign shape:((127, 1), (16, 256/16))  zN shape: ((16, Ceil(127/16)), (16, 256/16))
*/
template <class Element, class Layout, class Enable1 = void, class Enable2 = void>
struct iszNUnAlign {
    static bool const value = false;
};

template <class Element, class Layout>
struct iszNUnAlign<
    Element, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BytesToElements<Element>(Catlass::BYTE_PER_C0);
    static bool const value =
        (shape<0, 1>(Layout{}) == 1 && shape<1, 0>(Layout{}) == ELE_NUM_PER_C0 &&
         stride<0, 0>(Layout{}) == ELE_NUM_PER_C0 && stride<1, 0>(Layout{}) == 1);
};

template <class Element, class Layout, class Enable = void>
struct iszZ {
    static bool const value = false;
};

template <class Element, class Layout>
struct iszZ<Element, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BytesToElements<Element>(Catlass::BYTE_PER_C0);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL =
        Catlass::BytesToBits(Catlass::BYTE_PER_FRACTAL) / Catlass::SizeOfBits<Element>::value;
    static bool const value =
        (shape<0, 0>(Layout{}) == Catlass::C0_NUM_PER_FRACTAL && shape<1, 0>(Layout{}) == ELE_NUM_PER_C0 &&
         stride<1, 0>(Layout{}) == 1 && stride<1, 1>(Layout{}) == ELE_NUM_PER_FRACTAL);
};

template <class Element, class Layout, class Enable1 = void, class Enable2 = void>
struct isnZ {
    static bool const value = false;
};

template <class Element, class Layout>
struct isnZ<
    Element, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<rank_v<decltype(stride<0>(Layout{}))> == 2 && rank_v<decltype(stride<1>(Layout{}))> == 2>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = Catlass::BytesToElements<Element>(Catlass::BYTE_PER_C0);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL =
        Catlass::BytesToBits(Catlass::BYTE_PER_FRACTAL) / Catlass::SizeOfBits<Element>::value;
    static bool const value =
        (shape<0, 0>(Layout{}) == ELE_NUM_PER_C0 && shape<1, 0>(Layout{}) == Catlass::C0_NUM_PER_FRACTAL &&
         stride<0, 0>(Layout{}) == 1 && stride<1, 1>(Layout{}) == ELE_NUM_PER_FRACTAL);
};

#if (defined(CATLASS_ARCH) && (CATLASS_ARCH == 3510 || CATLASS_ARCH == 9201 || CATLASS_ARCH == 9202))
template <class Element, class Layout, class Enable1 = void, class Enable2 = void>
struct isMxScaleForRowMajorA {
    static bool const value = false;
};

template <class Layout>
struct isMxScaleForRowMajorA<
    float8_e8m0_t, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<
        rank_v<decltype(stride<0>(Layout{}))> == 1 && rank_v<decltype(stride<1>(Layout{}))> == 2 &&
        !is_constant<2, decltype(stride<0>(Layout{}))>::value &&
        ((rank_v<decltype(shape<0>(Layout{}))> == 1 && rank_v<decltype(shape<1>(Layout{}))> == 2) ||
         (rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2))>> {
    static bool const value = true;
};

template <class Element, class Layout, class Enable1 = void, class Enable2 = void>
struct isMxScaleForColumnMajorA {
    static bool const value = false;
};

template <class Layout>
struct isMxScaleForColumnMajorA<
    float8_e8m0_t, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<
        rank_v<decltype(stride<0>(Layout{}))> == 1 && rank_v<decltype(stride<1>(Layout{}))> == 2 &&
        is_constant<2, decltype(stride<0>(Layout{}))>::value &&
        ((rank_v<decltype(shape<0>(Layout{}))> == 1 && rank_v<decltype(shape<1>(Layout{}))> == 2) ||
         (rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2))>> {
    static bool const value = true;
};

template <class Element, class Layout, class Enable1 = void, class Enable2 = void, class Enable3 = void>
struct isMxScaleForRowMajorB {
    static bool const value = false;
};

template <class Layout>
struct isMxScaleForRowMajorB<
    float8_e8m0_t, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<rank_v<decltype(stride<0>(Layout{}))> == 2 && rank_v<decltype(stride<1>(Layout{}))> == 1>,
    std::enable_if_t<
        !is_constant<2, decltype(stride<0, 1>(Layout{}))>::value &&
        ((rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 1) ||
         (rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2))>> {
    static bool const value = true;
};

template <class Element, class Layout, class Enable1 = void, class Enable2 = void, class Enable3 = void>
struct isMxScaleForColumnMajorB {
    static bool const value = false;
};

template <class Layout>
struct isMxScaleForColumnMajorB<
    float8_e8m0_t, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<rank_v<decltype(stride<0>(Layout{}))> == 2 && rank_v<decltype(stride<1>(Layout{}))> == 1>,
    std::enable_if_t<
        is_constant<2, decltype(stride<0, 1>(Layout{}))>::value &&
        ((rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 1) ||
         (rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2))>> {
    static bool const value = true;
};

template <class Element, class Layout, class Enable1 = void, class Enable2 = void>
struct isMxScaleForzZ {
    static bool const value = false;
};

template <class Layout>
struct isMxScaleForzZ<
    float8_e8m0_t, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<
        rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2 &&
        rank_v<decltype(stride<0>(Layout{}))> == 2 && rank_v<decltype(stride<1>(Layout{}))> == 2>> {
    // TagToLayout zZ for e8m0 uses BYTE_PER_C0 (32) in shape, not MX MakeMxScaleLayout's 2; match by rank only.
    static bool const value = true;
};

template <class Element, class Layout, class Enable1 = void, class Enable2 = void>
struct isMxScaleFornN {
    static bool const value = false;
};

template <class Layout>
struct isMxScaleFornN<
    float8_e8m0_t, Layout, std::enable_if_t<Layout::depth == 2 && Layout::rank == 2>,
    std::enable_if_t<
        rank_v<decltype(shape<0>(Layout{}))> == 2 && rank_v<decltype(shape<1>(Layout{}))> == 2 &&
        rank_v<decltype(stride<0>(Layout{}))> == 2 && rank_v<decltype(stride<1>(Layout{}))> == 2>> {
    static bool const value = true;
};
#endif

} // end namespace detail

// Tile layout utilities
namespace detail {

template <class OriginBase, class TileShape, class Coord, int... Is>
CATLASS_HOST_DEVICE constexpr auto CropOriginShape(
    OriginBase const& originBase, TileShape const& tileShape, Coord const& coord, seq<Is...>)
{
    return MakeShape(tla::min(
        static_cast<uint32_t>(get<Is>(tileShape)),
        (static_cast<uint32_t>(get<Is>(coord)) < static_cast<uint32_t>(get<Is>(originBase))) ?
            (static_cast<uint32_t>(get<Is>(originBase)) - static_cast<uint32_t>(get<Is>(coord))) :
            0u)...);
}

} // namespace detail

/// 创建 tile layout：使用指定的 tile 尺寸用于内存布局计算，同时携带实际逻辑尺寸（origin_shape）。
/// coord 是元素坐标，用于计算实际的 originShape（处理边界情况）。
/// Supports layouts of any rank (rank >= 1) for depth==1 layouts.
/// For depth>1 (fractal) layouts, currently only rank-2 is supported.
template <class Layout, class TileShape, class Coord>
CATLASS_HOST_DEVICE constexpr auto GetTileLayout(Layout const& layout, TileShape const& tileShape, Coord const& coord)
{
    static_assert(
        is_tuple<TileShape>::value && depth_v<TileShape> == 1 && rank_v<TileShape> >= 1,
        "GetTileLayout: TileShape must be a flat tuple with rank >= 1.");
    static_assert(
        is_tuple<Coord>::value && depth_v<Coord> == 1 && rank_v<Coord> == rank_v<TileShape>,
        "GetTileLayout: Coord must have the same rank as TileShape.");

    // 统一计算 tail tile 的逻辑尺寸（originShape 裁剪）
    auto tileOriginShape = detail::CropOriginShape(layout.originShape(), tileShape, coord, tuple_seq<TileShape>{});

    // depth==1 的布局（vector/matrix/tensor）：tile shape 直接作为 memory-layout shape
    // 支持任意 rank >= 1（但必须与 layout.rank 匹配）
    if constexpr (Layout::depth == 1) {
        static_assert(
            Layout::rank == rank_v<TileShape>,
            "GetTileLayout: for depth==1 layouts, TileShape rank must match layout rank.");
        return MakeLayout(tileShape, layout.stride(), tileOriginShape);
    } else {
        // depth>1 的布局（fractal layout）：目前只支持 rank=2
        // 因为 fractal layout 通常用于矩阵（rank-2），需要把 (rows, cols) 转为同结构嵌套 shape
        static_assert(
            rank_v<TileShape> == 2,
            "GetTileLayout: for depth>1 (fractal) layouts, TileShape must be rank-2 (rows, cols).");

        const uint32_t rows = get<0>(tileShape);
        const uint32_t cols = get<1>(tileShape);

        // MakeMxScaleLayout RowMajor A 等：第一维为行长度，第二维为 (C0, ceil(cols/C0))；与 catlass_dev
        // `MakeLayoutTile` 中 rank(shape<0>)==1 && rank(shape<1>)==2 分支一致。
        if constexpr (
            Layout::depth == 2 && Layout::rank == 2 && rank_v<decltype(shape<0>(Layout{}))> == 1 &&
            rank_v<decltype(shape<1>(Layout{}))> == 2 && is_static<decltype(shape<1, 0>(layout))>::value) {
            constexpr uint32_t ELE_NUM_PER_C0 = decltype(shape<1, 0>(layout))::value;
            return MakeLayout(
                MakeShape(rows, MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(cols, Int<ELE_NUM_PER_C0>{}))),
                layout.stride(), tileOriginShape);
        }
        // MakeMxScaleLayout B 侧等：shape 为 ((C0, ceil(rows/C0)), cols)；与 catlass_dev `MakeLayoutTile` 中
        // rank(shape<0>)==2 && rank(shape<1>)==1 分支一致。
        else if constexpr (
            Layout::depth == 2 && Layout::rank == 2 && rank_v<decltype(shape<0>(Layout{}))> == 2 &&
            rank_v<decltype(shape<1>(Layout{}))> == 1 && is_static<decltype(shape<0, 0>(layout))>::value) {
            constexpr uint32_t ELE_NUM_PER_C0 = decltype(shape<0, 0>(layout))::value;
            return MakeLayout(
                MakeShape(MakeShape(Int<ELE_NUM_PER_C0>{}, CeilDiv(rows, Int<ELE_NUM_PER_C0>{})), cols),
                layout.stride(), tileOriginShape);
        }
        // 典型 fractal（zN/nZ 等）：shape<0,0>、shape<1,0> 均为编译期常量；嵌套 ((r0,...),(c0,...))。
        else if constexpr (
            is_static<decltype(shape<0, 0>(layout))>::value && is_static<decltype(shape<1, 0>(layout))>::value) {
            constexpr uint32_t dstInnerShapeRow = decltype(shape<0, 0>(layout))::value;
            constexpr uint32_t dstInnerShapeCol = decltype(shape<1, 0>(layout))::value;
            return MakeLayout(
                MakeShape(
                    MakeShape(Int<dstInnerShapeRow>{}, CeilDiv<dstInnerShapeRow>(rows)),
                    MakeShape(Int<dstInnerShapeCol>{}, CeilDiv<dstInnerShapeCol>(cols))),
                layout.stride(), tileOriginShape);
        }
        // 内层块尺寸非编译期常量：运行期从 layout 读取再分块。
        else {
            const uint32_t dstInnerShapeRow = shape<0, 0>(layout);
            const uint32_t dstInnerShapeCol = shape<1, 0>(layout);
            return MakeLayout(
                MakeShape(
                    MakeShape(dstInnerShapeRow, CeilDiv(rows, dstInnerShapeRow)),
                    MakeShape(dstInnerShapeCol, CeilDiv(cols, dstInnerShapeCol))),
                layout.stride(), tileOriginShape);
        }
    }
}

// Layout transforms
namespace detail {

// Prepend one leading dimension to a layout type (general form of "make batched layout").
// Preserves the *types* of each stride element from the original layout.
template <
    class Layout, class NewShapeT = uint32_t, class NewStrideT = int64_t, class NewOriginT = uint32_t,
    class Seq = tla::make_seq<Layout::rank>>
struct PrependDimLayout;

template <class Layout, class NewShapeT, class NewStrideT, class NewOriginT, int... Is>
struct PrependDimLayout<Layout, NewShapeT, NewStrideT, NewOriginT, tla::seq<Is...>> {
    using ShapeOld = tla::remove_cvref_t<decltype(std::declval<Layout const&>().shape())>;
    using StrideOld = tla::remove_cvref_t<decltype(std::declval<Layout const&>().stride())>;
    using OriginOld = tla::remove_cvref_t<decltype(std::declval<Layout const&>().originShape())>;

    using ShapeNew = tla::Shape<NewShapeT, tla::remove_cvref_t<decltype(tla::get<Is>(std::declval<ShapeOld>()))>...>;
    using StrideNew =
        tla::Stride<NewStrideT, tla::remove_cvref_t<decltype(tla::get<Is>(std::declval<StrideOld>()))>...>;
    using OriginNew = tla::Shape<NewOriginT, tla::remove_cvref_t<decltype(tla::get<Is>(std::declval<OriginOld>()))>...>;

    using type = tla::Layout<ShapeNew, StrideNew, OriginNew>;
};

} // namespace detail

template <class Layout, class NewShapeT = uint32_t, class NewStrideT = int64_t, class NewOriginT = uint32_t>
using PrependDimLayout_t = typename detail::PrependDimLayout<Layout, NewShapeT, NewStrideT, NewOriginT>::type;

template <class Layout>
using MakeBatchedLayout_t = PrependDimLayout_t<Layout>;

} // end namespace tla

#endif // TLA_LAYOUT_HPP
