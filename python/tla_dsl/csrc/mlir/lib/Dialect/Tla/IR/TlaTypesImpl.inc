// Custom parsers/printers for Tla TypeDefs. Storage and uniquing live in
// TlaDialect.cpp (GET_TYPEDEF_CLASSES + Types.cpp.inc).
#ifndef TLA_DIALECT_TLA_IR_TLATYPESIMPL_INC
#define TLA_DIALECT_TLA_IR_TLATYPESIMPL_INC

#include "Dialect/Tla/IR/TlaTypes.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace {
constexpr int64_t kTreeOpen = -2;
constexpr int64_t kTreeClose = -3;

ParseResult parseIndexTreeLeaf(AsmParser &parser, SmallVectorImpl<int64_t> &tree) {
  if (succeeded(parser.parseOptionalQuestion())) {
    tree.push_back(ShapedType::kDynamic);
    return success();
  }

  int64_t value = 0;
  if (parser.parseInteger(value))
    return failure();
  if (value < 0)
    return parser.emitError(parser.getCurrentLocation())
           << "cat index-tree leaves must be non-negative integers or ?";
  tree.push_back(value);
  return success();
}

ParseResult parseIndexTreeLeafList(AsmParser &parser, SmallVectorImpl<int64_t> &tree) {
  if (parseIndexTreeLeaf(parser, tree))
    return failure();
  while (succeeded(parser.parseOptionalComma())) {
    if (parseIndexTreeLeaf(parser, tree))
      return failure();
  }
  return success();
}

ParseResult parseIndexTreeElement(AsmParser &parser, SmallVectorImpl<int64_t> &tree) {
  if (succeeded(parser.parseOptionalLParen())) {
    tree.push_back(kTreeOpen);
    if (parseIndexTreeLeafList(parser, tree) || parser.parseRParen())
      return failure();
    tree.push_back(kTreeClose);
    return success();
  }
  return parseIndexTreeLeaf(parser, tree);
}

ParseResult parseIndexTreeElementList(AsmParser &parser, SmallVectorImpl<int64_t> &tree) {
  if (parseIndexTreeElement(parser, tree))
    return failure();
  while (succeeded(parser.parseOptionalComma())) {
    if (parseIndexTreeElement(parser, tree))
      return failure();
  }
  return success();
}

LogicalResult verifyIndexTree(ArrayRef<int64_t> tree) {
  if (tree.empty())
    return failure();
  int64_t depth = 0;
  int64_t currentGroupLeaves = 0;
  bool sawLeaf = false;
  for (int64_t token : tree) {
    if (token == kTreeOpen) {
      if (depth != 0)
        return failure();
      ++depth;
      currentGroupLeaves = 0;
      continue;
    }
    if (token == kTreeClose) {
      if (depth != 1 || currentGroupLeaves == 0)
        return failure();
      --depth;
      continue;
    }
    if (token < 0 && token != ShapedType::kDynamic)
      return failure();
    if (depth == 1)
      ++currentGroupLeaves;
    sawLeaf = true;
  }
  return success(sawLeaf && depth == 0);
}

void printIndexTree(AsmPrinter &printer, ArrayRef<int64_t> tree) {
  SmallVector<bool, 4> firstStack;
  firstStack.push_back(true);
  auto printSeparatorForItem = [&] {
    if (!firstStack.back())
      printer << ",";
    firstStack.back() = false;
  };

  for (int64_t token : tree) {
    if (token == kTreeOpen) {
      printSeparatorForItem();
      printer << "(";
      firstStack.push_back(true);
      continue;
    }
    if (token == kTreeClose) {
      printer << ")";
      if (firstStack.size() > 1)
        firstStack.pop_back();
      continue;
    }
    printSeparatorForItem();
    if (token == ShapedType::kDynamic)
      printer << "?";
    else
      printer << token;
  }
}

LogicalResult flattenIndexTree(ArrayRef<int64_t> tree, SmallVectorImpl<int64_t> &leaves) {
  leaves.clear();
  if (failed(verifyIndexTree(tree)))
    return failure();
  for (int64_t token : tree) {
    if (token == kTreeOpen || token == kTreeClose)
      continue;
    leaves.push_back(token);
  }
  return success(!leaves.empty());
}

bool sameIndexTreeStructure(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
  if (failed(verifyIndexTree(lhs)) || failed(verifyIndexTree(rhs)) || lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip_equal(lhs, rhs)) {
    if ((left == kTreeOpen) != (right == kTreeOpen) ||
        (left == kTreeClose) != (right == kTreeClose))
      return false;
  }
  return true;
}

template <typename TypeT> Type parseIndexTreeType(AsmParser &parser, StringRef mnemonic) {
  SmallVector<int64_t, 8> tree;
  if (parser.parseLess() || parseIndexTreeElementList(parser, tree) || parser.parseGreater())
    return (parser.emitError(parser.getCurrentLocation())
                << "invalid tla." << mnemonic << " index-tree",
            Type{});
  return TypeT::getChecked(
      [&] { return parser.emitError(parser.getCurrentLocation()) << "invalid tla." << mnemonic; },
      parser.getContext(), tree);
}

FailureOr<AddressSpace> parseAddressSpace(StringRef text) {
  if (auto sym = symbolizeAddressSpace(text))
    return *sym;
  return failure();
}

Type parseTypeFromText(AsmParser &parser, StringRef text) {
  Type type = parseType(text, parser.getContext());
  if (!type)
    parser.emitError(parser.getCurrentLocation()) << "invalid MLIR type: " << text;
  return type;
}

} // namespace

namespace tla {

LogicalResult MaskSSAType::verify(function_ref<InFlightDiagnostic()> emitError,
                               int64_t physicalLanes) {
  if (physicalLanes != 32 && physicalLanes != 64 && physicalLanes != 128 && physicalLanes != 256)
    return emitError()
           << "tla.mask physical lane count must be one of 32, 64, 128, or 256, got "
           << physicalLanes;
  return success();
}

LogicalResult getTlaIndexTreeLeaves(ArrayRef<int64_t> tree, SmallVectorImpl<int64_t> &leaves) {
  return flattenIndexTree(tree, leaves);
}

Type ShapeType::parse(AsmParser &parser) { return parseIndexTreeType<ShapeType>(parser, "shape"); }
void ShapeType::print(AsmPrinter &printer) const {
  printer << "<";
  printIndexTree(printer, getTree());
  printer << ">";
}
LogicalResult ShapeType::verify(function_ref<InFlightDiagnostic()> emitError,
                                ArrayRef<int64_t> tree) {
  if (failed(verifyIndexTree(tree)))
    return emitError() << "invalid tla.shape index-tree";
  return success();
}

Type CoordType::parse(AsmParser &parser) { return parseIndexTreeType<CoordType>(parser, "coord"); }
void CoordType::print(AsmPrinter &printer) const {
  printer << "<";
  printIndexTree(printer, getTree());
  printer << ">";
}
LogicalResult CoordType::verify(function_ref<InFlightDiagnostic()> emitError,
                                ArrayRef<int64_t> tree) {
  if (failed(verifyIndexTree(tree)))
    return emitError() << "invalid tla.coord index-tree";
  return success();
}

Type StrideType::parse(AsmParser &parser) {
  return parseIndexTreeType<StrideType>(parser, "stride");
}
void StrideType::print(AsmPrinter &printer) const {
  printer << "<";
  printIndexTree(printer, getTree());
  printer << ">";
}
LogicalResult StrideType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<int64_t> tree) {
  if (failed(verifyIndexTree(tree)))
    return emitError() << "invalid tla.stride index-tree";
  return success();
}

LogicalResult LayoutType::verify(function_ref<InFlightDiagnostic()> emitError, ShapeType shape,
                                 StrideType stride, ShapeType origin, LayoutTag layoutTag) {
  if (!shape || !stride || !origin)
    return emitError() << "tla.layout shape, stride, and origin must be non-null";
  SmallVector<int64_t, 8> shapeLeaves;
  SmallVector<int64_t, 8> strideLeaves;
  if (failed(flattenIndexTree(shape.getTree(), shapeLeaves)) ||
      failed(flattenIndexTree(stride.getTree(), strideLeaves)) ||
      shapeLeaves.size() != strideLeaves.size())
    return emitError() << "tla.layout shape and stride must have the same leaf count";
  if (!sameIndexTreeStructure(shape.getTree(), stride.getTree()))
    return emitError() << "tla.layout shape and stride must have the same group structure";
  SmallVector<int64_t, 8> originLeaves;
  if (failed(flattenIndexTree(origin.getTree(), originLeaves)))
    return emitError() << "tla.layout origin must be valid and non-empty";
  // Linear layouts (RowMajor / ColumnMajor / Vector) are non-fractal, so their
  // origin shape mirrors the shape leaves one for one.
  if (::tla::isLinearLayout(layoutTag) && originLeaves.size() != shapeLeaves.size())
    return emitError() << "tla.layout linear origin must match shape leaf count";
  // Vector is the one-dimensional linear tag: a flat, unit-stride run of
  // elements. Spelling it with more than one leaf would silently address only
  // the leading index, so it is rejected here rather than in each consumer.
  if (::tla::isVectorLayout(layoutTag) && shapeLeaves.size() != 1)
    return emitError() << "tla.layout Vector shape must be one-dimensional";
  return success();
}

LogicalResult TlaTensorType::verify(function_ref<InFlightDiagnostic()> emitError, LayoutType layout,
                                    CoordType coord, PtrType ptr) {
  if (!layout || !coord || !ptr)
    return emitError() << "tla.tensor layout/coord/ptr fields must be non-null";
  if (!ptr.getPointee())
    return emitError() << "tla.tensor pointer pointee type must be non-null";
  SmallVector<int64_t, 8> shapeLeaves;
  SmallVector<int64_t, 8> strideLeaves;
  SmallVector<int64_t, 8> coordLeaves;
  SmallVector<int64_t, 8> originLeaves;
  if (failed(flattenIndexTree(layout.getShape().getTree(), shapeLeaves)) ||
      failed(flattenIndexTree(layout.getStride().getTree(), strideLeaves)) ||
      failed(flattenIndexTree(coord.getTree(), coordLeaves)) ||
      failed(flattenIndexTree(layout.getOrigin().getTree(), originLeaves)))
    return emitError() << "tla.tensor index-tree fields must be valid and non-empty";
  if (shapeLeaves.size() != strideLeaves.size())
    return emitError() << "tla.tensor layout shape and stride must have the same leaf count";
  if (coordLeaves.size() != originLeaves.size())
    return emitError() << "tla.tensor coord and origin must have the same leaf count";
  return success();
}

Type VectorSSAType::parse(AsmParser &parser) {
  auto &dialectParser = static_cast<DialectAsmParser &>(parser);
  StringRef full = dialectParser.getFullSymbolSpec();
  if (!full.consume_front("vector<") || !full.consume_back(">"))
    return (parser.emitError(parser.getCurrentLocation(),
                             "expected !tla.vector<NxT>"),
            Type{});

  auto [lanesText, elementText] = full.split('x');
  if (lanesText.empty() || elementText.empty())
    return (parser.emitError(parser.getCurrentLocation(),
                             "expected !tla.vector<NxT>"),
            Type{});

  int64_t validLanes = ShapedType::kDynamic;
  if (lanesText != "?" && lanesText.getAsInteger(10, validLanes))
    return (parser.emitError(parser.getCurrentLocation(),
                             "expected vector lane count as a positive integer or ?"),
            Type{});

  Type elementType = parseTypeFromText(parser, elementText);
  if (!elementType)
    return Type{};
  return VectorSSAType::getChecked(
      [&] { return parser.emitError(parser.getCurrentLocation())
                   << "invalid tla.vector type"; },
      parser.getContext(), validLanes, elementType);
}

void VectorSSAType::print(AsmPrinter &printer) const {
  printer << "<";
  if (getValidLanes() == ShapedType::kDynamic)
    printer << "?";
  else
    printer << getValidLanes();
  printer << "x" << getElementType() << ">";
}

LogicalResult VectorSSAType::verify(function_ref<InFlightDiagnostic()> emitError,
                                    int64_t validLanes, Type elementType) {
  if (!elementType)
    return emitError() << "tla.vector element type must be non-null";
  if (!mlir::isa<IntegerType, FloatType, HiFloat8Type, Float4E2M1x2Type, Float4E1M2x2Type>(elementType))
    return emitError() << "tla.vector element type must be an integer or float, got "
                       << elementType;

  unsigned bitWidth = getBitSizeOfFixedWidthScalarType(elementType);
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return emitError() << "tla.vector element type must have a byte-aligned width, got "
                       << elementType;

  constexpr int64_t kVectorRegisterBytes = 256;
  int64_t capacity = kVectorRegisterBytes / static_cast<int64_t>(bitWidth / 8);
  if (capacity <= 0)
    return emitError() << "tla.vector element type is wider than the 256-byte register";
  if (validLanes != ShapedType::kDynamic &&
      (validLanes <= 0 || validLanes > capacity))
    return emitError() << "tla.vector valid lane count must be in [1, " << capacity
                       << "] for element type " << elementType << ", got " << validLanes;
  return success();
}

} // namespace tla

#endif // TLA_DIALECT_TLA_IR_TLATYPESIMPL_INC
