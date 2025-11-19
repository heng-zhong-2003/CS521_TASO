namespace xla {
namespace {
absl::Status TASOVisitor::HandleAll(HloInstruction *root) {

  HloInstruction *str0, *str1, *str2, *str2, *str3, *str4;

          // Match source pattern
          if (Match(root, m::Add(m::Op(&str0), m::Op(&str1)))) {
        

  // Create replacement pattern
    HloInstruction* str5 = root->AddInstruction(HloInstruction::CreateBinary(root->shape(), HloOpcode::Add, str1, str0));
    HloInstruction* str6 = root->AddInstruction(HloInstruction::CreateBinary(root->shape(), HloOpcode::Add, str1, str0));
    return ReplaceWithNewInstruction(root, HloInstruction::CreateBinary(root->shape(), HloOpcode::Add, str1, str0));
  }

}
}
return absl::OkStatus();
}

namespace xla {
namespace {
absl::Status TASOVisitor::HandleAll(HloInstruction *root) {

  HloInstruction *str0, *str1, *str2, *str2, *str3, *str4;

          // Match source pattern
          if (Match(root, m::Add(m::Op(&str0), m::Op(&str1)))) {
        

  // Create replacement pattern
    HloInstruction* str5 = root->AddInstruction(HloInstruction::CreateBinary(root->shape(), HloOpcode::Dot, str0, str1));
    HloInstruction* str6 = root->AddInstruction(HloInstruction::CreateBinary(root->shape(), HloOpcode::Dot, str0, str1));
    return ReplaceWithNewInstruction(root, HloInstruction::CreateBinary(root->shape(), HloOpcode::Dot, str0, str1));
  }

}
}
return absl::OkStatus();
}

namespace xla {
namespace {
absl::Status TASOVisitor::HandleAll(HloInstruction *root) {

  HloInstruction *str0, *str1, *str2, *str2, *str3, *str4;

          // Match source pattern
          if (Match(root, m::Add(m::Op(&str0), m::Op(&str1)))) {
        

  // Create replacement pattern
    HloInstruction* str5 = root->AddInstruction(HloInstruction::CreateBinary(root->shape(), HloOpcode::Dot, str1, str0));
    HloInstruction* str6 = root->AddInstruction(HloInstruction::CreateBinary(root->shape(), HloOpcode::Dot, str1, str0));
    return ReplaceWithNewInstruction(root, HloInstruction::CreateBinary(root->shape(), HloOpcode::Dot, str1, str0));
  }

}
}
return absl::OkStatus();
}

namespace xla {
namespace {
absl::Status TASOVisitor::HandleAll(HloInstruction *root) {

  HloInstruction *str0, *str1, *str2, *str2, *str3, *str4;

          // Match source pattern
          if (Match(root, m::Add(m::Op(&str1), m::Op(&str0)))) {
        

  // Create replacement pattern
    HloInstruction* str5 = root->AddInstruction(HloInstruction::CreateBinary(root->shape(), HloOpcode::Dot, str0, str1));
    HloInstruction* str6 = root->AddInstruction(HloInstruction::CreateBinary(root->shape(), HloOpcode::Dot, str0, str1));
    return ReplaceWithNewInstruction(root, HloInstruction::CreateBinary(root->shape(), HloOpcode::Dot, str0, str1));
  }

}
}
return absl::OkStatus();
}

namespace xla {
namespace {
absl::Status TASOVisitor::HandleAll(HloInstruction *root) {

  HloInstruction *str0, *str1, *str2, *str2, *str3, *str4;

          // Match source pattern
          if (Match(root, m::Add(m::Op(&str1), m::Op(&str0)))) {
        

  // Create replacement pattern
    HloInstruction* str5 = root->AddInstruction(HloInstruction::CreateBinary(root->shape(), HloOpcode::Dot, str1, str0));
    HloInstruction* str6 = root->AddInstruction(HloInstruction::CreateBinary(root->shape(), HloOpcode::Dot, str1, str0));
    return ReplaceWithNewInstruction(root, HloInstruction::CreateBinary(root->shape(), HloOpcode::Dot, str1, str0));
  }

}
}
return absl::OkStatus();
}

namespace xla {
namespace {
absl::Status TASOVisitor::HandleAll(HloInstruction *root) {

  HloInstruction *str0, *str1, *str2, *str2, *str3, *str4;

          // Match source pattern
          if (Match(root, m::Dot(m::Op(&str0), m::Op(&str1)))) {
        

  // Create replacement pattern
    HloInstruction* str5 = root->AddInstruction(HloInstruction::CreateBinary(root->shape(), HloOpcode::Dot, str1, str0));
    HloInstruction* str6 = root->AddInstruction(HloInstruction::CreateBinary(root->shape(), HloOpcode::Dot, str1, str0));
    return ReplaceWithNewInstruction(root, HloInstruction::CreateBinary(root->shape(), HloOpcode::Dot, str1, str0));
  }

}
}
return absl::OkStatus();
}

